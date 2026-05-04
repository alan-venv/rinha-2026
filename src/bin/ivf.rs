use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::thread;

use rinha::*;

use anyhow::{Result, bail};
use memmap2::Mmap;

const REFERENCES_PATH: &str = "resources/references.bin";
const PRIMARY_IVF_PATH: &str = "resources/ivf.bin";
const FRAUD_IVF_PATH: &str = "resources/ivf-fraud.bin";
const LEGIT_IVF_PATH: &str = "resources/ivf-legit.bin";
const SAMPLE_MULTIPLIER: usize = 64;
const PQ_SAMPLE_MULTIPLIER: usize = 64;

type PqSubVector = [i16; PQ_DIMENSIONS_PER_SUBQUANTIZER];

fn main() -> Result<()> {
    build_all_indexes()
}

pub(crate) fn build_all_indexes() -> Result<()> {
    validate_shared_config()?;

    let references = ReferenceDataset::load(Path::new(REFERENCES_PATH))?;
    let workers = worker_count();

    build_primary_index(&references, workers)?;
    build_label_index(
        "fraud",
        &references,
        true,
        Path::new(FRAUD_IVF_PATH),
        workers,
    )?;
    build_label_index(
        "legit",
        &references,
        false,
        Path::new(LEGIT_IVF_PATH),
        workers,
    )?;

    Ok(())
}

fn validate_shared_config() -> Result<()> {
    validate_non_zero("centroid count", IVF_CENTROIDS)?;
    validate_non_zero("auxiliary centroid count", IVF_AUX_CENTROIDS)?;
    validate_non_zero("assignments per reference", IVF_ASSIGNMENTS_PER_REFERENCE)?;
    validate_non_zero("PQ subquantizers", PQ_SUBQUANTIZERS)?;
    validate_non_zero("PQ codewords", PQ_CODEWORDS)?;
    validate_non_zero(
        "PQ dimensions per subquantizer",
        PQ_DIMENSIONS_PER_SUBQUANTIZER,
    )?;

    if PQ_SUBQUANTIZERS * PQ_DIMENSIONS_PER_SUBQUANTIZER != VECTOR_DIMENSIONS {
        bail!(
            "invalid PQ layout: {}x{} does not cover {} dimensions",
            PQ_SUBQUANTIZERS,
            PQ_DIMENSIONS_PER_SUBQUANTIZER,
            VECTOR_DIMENSIONS
        );
    }

    if PQ_CODEWORDS > u8::MAX as usize + 1 {
        bail!("invalid PQ codewords: {} > 256", PQ_CODEWORDS);
    }

    if IVF_ASSIGNMENTS_PER_REFERENCE > IVF_CENTROIDS {
        bail!(
            "invalid assignments per reference for primary IVF: {} > {}",
            IVF_ASSIGNMENTS_PER_REFERENCE,
            IVF_CENTROIDS
        );
    }

    if IVF_ASSIGNMENTS_PER_REFERENCE > IVF_AUX_CENTROIDS {
        bail!(
            "invalid assignments per reference for auxiliary IVF: {} > {}",
            IVF_ASSIGNMENTS_PER_REFERENCE,
            IVF_AUX_CENTROIDS
        );
    }

    Ok(())
}

fn validate_non_zero(name: &str, value: usize) -> Result<()> {
    if value == 0 {
        bail!("invalid {name}: {value}");
    }

    Ok(())
}

fn build_primary_index(references: &ReferenceDataset, workers: usize) -> Result<()> {
    let all_references = AllReferences::new(references);
    build_index(
        "all",
        &all_references,
        references.len(),
        IVF_CENTROIDS,
        Path::new(PRIMARY_IVF_PATH),
        workers,
    )
}

fn build_label_index(
    name: &str,
    references: &ReferenceDataset,
    is_fraud: bool,
    output: &Path,
    workers: usize,
) -> Result<()> {
    let label_references = LabelReferences::new(references, is_fraud);
    build_index(
        name,
        &label_references,
        references.len(),
        IVF_AUX_CENTROIDS,
        output,
        workers,
    )
}

fn validate_centroid_count(
    name: &str,
    references: &impl ReferenceView,
    centroid_count: usize,
) -> Result<()> {
    if centroid_count > references.len() {
        bail!(
            "invalid centroid count for {name} references: {} > {}",
            centroid_count,
            references.len()
        );
    }

    Ok(())
}

fn build_index(
    name: &str,
    references: &impl ReferenceView,
    total_reference_count: usize,
    centroid_count: usize,
    output: &Path,
    workers: usize,
) -> Result<()> {
    validate_centroid_count(name, references, centroid_count)?;

    let sample_count = SAMPLE_MULTIPLIER * centroid_count;

    println!(
        "building {name} IVF_PQ: references={}, centroids={}, assignments_per_reference={}, sample={}, pq={}x{}, codewords={}, iterations={}, workers={}",
        references.len(),
        centroid_count,
        IVF_ASSIGNMENTS_PER_REFERENCE,
        sample_count,
        PQ_SUBQUANTIZERS,
        PQ_DIMENSIONS_PER_SUBQUANTIZER,
        PQ_CODEWORDS,
        IVF_ITERATIONS,
        workers,
    );

    let centroids = train_centroids(
        references,
        centroid_count,
        sample_count,
        IVF_ITERATIONS,
        workers,
    );
    let assignments = assign_references(references, &centroids, workers);
    let codebooks =
        train_pq_codebooks(references, &centroids, &assignments.assignments_by_position);
    let (indices, codes) = candidate_indices_and_codes(
        references,
        &centroids,
        &codebooks,
        &assignments.offsets,
        assignments.assignments_by_position,
    );
    write_index(
        output,
        total_reference_count,
        &centroids,
        &codebooks,
        &assignments.offsets,
        &indices,
        &codes,
    )?;

    println!(
        "wrote {}: {} bytes",
        output.display(),
        IVF_HEADER_LEN
            + centroids.len() * VECTOR_LEN
            + codebooks.len() * PQ_DIMENSIONS_PER_SUBQUANTIZER * size_of::<i16>()
            + assignments.offsets.len() * size_of::<u32>()
            + indices.len() * size_of::<u32>()
            + codes.len()
    );

    Ok(())
}

fn worker_count() -> usize {
    env::var("IVF_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or_else(|| thread::available_parallelism().map_or(1, usize::from))
}

fn train_centroids(
    references: &impl ReferenceView,
    centroid_count: usize,
    sample_count: usize,
    iterations: usize,
    workers: usize,
) -> Vec<ReferenceVector> {
    let samples = sample_references(references, sample_count.max(centroid_count));
    let mut centroids = initial_centroids(&samples, centroid_count);
    let workers = workers.min(samples.len().max(1));

    for iteration in 0..iterations {
        let chunk_len = samples.len().div_ceil(workers);
        let partials = thread::scope(|scope| {
            let mut tasks = Vec::with_capacity(workers);

            for chunk in samples.chunks(chunk_len) {
                let centroids = &centroids;
                tasks.push(scope.spawn(move || {
                    let mut sums = vec![[0_i64; VECTOR_DIMENSIONS]; centroid_count];
                    let mut counts = vec![0_u32; centroid_count];

                    for sample in chunk {
                        let centroid = nearest_centroid(centroids, sample);
                        counts[centroid] += 1;

                        for (sum, value) in sums[centroid].iter_mut().zip(sample) {
                            *sum += i64::from(*value);
                        }
                    }

                    (sums, counts)
                }));
            }

            tasks
                .into_iter()
                .map(|task| task.join().expect("worker thread panicked"))
                .collect::<Vec<_>>()
        });

        let mut sums = vec![[0_i64; VECTOR_DIMENSIONS]; centroid_count];
        let mut counts = vec![0_u32; centroid_count];

        for (partial_sums, partial_counts) in partials {
            for centroid in 0..centroid_count {
                counts[centroid] += partial_counts[centroid];

                for dimension in 0..VECTOR_DIMENSIONS {
                    sums[centroid][dimension] += partial_sums[centroid][dimension];
                }
            }
        }

        for centroid in 0..centroid_count {
            let count = counts[centroid];

            if count == 0 {
                continue;
            }

            for dimension in 0..VECTOR_DIMENSIONS {
                centroids[centroid][dimension] =
                    (sums[centroid][dimension] / i64::from(count)) as i16;
            }
        }

        println!("k-means iteration {} done", iteration + 1);
    }

    centroids
}

fn sample_references(
    references: &impl ReferenceView,
    requested_count: usize,
) -> Vec<ReferenceVector> {
    let sample_count = requested_count.min(references.len());
    let mut samples = Vec::with_capacity(sample_count);

    for sample in 0..sample_count {
        let index = sample * references.len() / sample_count;
        samples.push(references.vector_at(index));
    }

    samples
}

fn initial_centroids(samples: &[ReferenceVector], centroid_count: usize) -> Vec<ReferenceVector> {
    let mut centroids = Vec::with_capacity(centroid_count);

    for centroid in 0..centroid_count {
        let index = centroid * samples.len() / centroid_count;
        centroids.push(samples[index]);
    }

    centroids
}

fn assign_references(
    references: &impl ReferenceView,
    centroids: &[ReferenceVector],
    workers: usize,
) -> Assignments {
    let workers = workers.min(references.len().max(1));
    let chunk_len = references.len().div_ceil(workers);
    let chunks = thread::scope(|scope| {
        let mut tasks = Vec::with_capacity(workers);

        for chunk_start in (0..references.len()).step_by(chunk_len) {
            let chunk_end = (chunk_start + chunk_len).min(references.len());
            tasks.push(scope.spawn(move || {
                let mut counts = vec![0_u32; centroids.len()];
                let mut assignments = Vec::with_capacity(chunk_end - chunk_start);

                for position in chunk_start..chunk_end {
                    let vector = references.vector_at(position);
                    let nearest =
                        nearest_centroids::<IVF_ASSIGNMENTS_PER_REFERENCE>(centroids, &vector);
                    assignments.push(nearest.map(|centroid| centroid as u32));

                    for centroid in nearest {
                        counts[centroid] += 1;
                    }
                }

                AssignmentChunk {
                    start: chunk_start,
                    assignments,
                    centroid_counts: counts,
                }
            }));
        }

        tasks
            .into_iter()
            .map(|task| task.join().expect("worker thread panicked"))
            .collect::<Vec<_>>()
    });

    let mut ordered_chunks = chunks;
    ordered_chunks.sort_unstable_by_key(|chunk| chunk.start);

    let mut assignments_by_position = Vec::with_capacity(references.len());
    let mut centroid_counts = vec![0_u32; centroids.len()];

    for chunk in ordered_chunks {
        println!(
            "assigned {} references",
            (chunk.start + chunk.assignments.len()).min(references.len())
        );

        assignments_by_position.extend(chunk.assignments);

        for centroid in 0..centroids.len() {
            centroid_counts[centroid] += chunk.centroid_counts[centroid];
        }
    }

    let offsets = candidate_list_offsets(&centroid_counts);

    Assignments {
        offsets,
        assignments_by_position,
    }
}

struct Assignments {
    offsets: Vec<u32>,
    assignments_by_position: Vec<[u32; IVF_ASSIGNMENTS_PER_REFERENCE]>,
}

struct AssignmentChunk {
    start: usize,
    assignments: Vec<[u32; IVF_ASSIGNMENTS_PER_REFERENCE]>,
    centroid_counts: Vec<u32>,
}

fn candidate_list_offsets(centroid_counts: &[u32]) -> Vec<u32> {
    let mut offsets = vec![0_u32; centroid_counts.len() + 1];

    for centroid in 0..centroid_counts.len() {
        offsets[centroid + 1] = offsets[centroid] + centroid_counts[centroid];
    }

    offsets
}

fn candidate_indices_and_codes(
    references: &impl ReferenceView,
    centroids: &[ReferenceVector],
    codebooks: &[PqSubVector],
    offsets: &[u32],
    assignments_by_position: Vec<[u32; IVF_ASSIGNMENTS_PER_REFERENCE]>,
) -> (Vec<u32>, Vec<u8>) {
    let mut cursors = offsets[..offsets.len() - 1].to_vec();
    let entry_count = references.len() * IVF_ASSIGNMENTS_PER_REFERENCE;
    let mut indices = vec![0_u32; entry_count];
    let mut codes = vec![0_u8; entry_count * PQ_CODE_LEN];

    for (position, assigned_centroids) in assignments_by_position.into_iter().enumerate() {
        let reference_index = references.index_at(position);
        let vector = references.vector_at(position);

        for centroid in assigned_centroids {
            let cursor = &mut cursors[centroid as usize];
            let entry = *cursor as usize;
            indices[entry] = reference_index as u32;
            encode_residual(
                &vector,
                &centroids[centroid as usize],
                codebooks,
                &mut codes[entry * PQ_CODE_LEN..(entry + 1) * PQ_CODE_LEN],
            );
            *cursor += 1;
        }
    }

    (indices, codes)
}

fn train_pq_codebooks(
    references: &impl ReferenceView,
    centroids: &[ReferenceVector],
    assignments_by_position: &[[u32; IVF_ASSIGNMENTS_PER_REFERENCE]],
) -> Vec<PqSubVector> {
    let requested = PQ_SAMPLE_MULTIPLIER * PQ_CODEWORDS;
    let samples = sample_residuals(
        references,
        centroids,
        assignments_by_position,
        requested.max(PQ_CODEWORDS),
    );
    let mut codebooks = Vec::with_capacity(PQ_SUBQUANTIZERS * PQ_CODEWORDS);

    for subquantizer in 0..PQ_SUBQUANTIZERS {
        println!(
            "training PQ subquantizer {}/{} with {} samples",
            subquantizer + 1,
            PQ_SUBQUANTIZERS,
            samples.len()
        );
        let mut codebook = initial_pq_codebook(&samples, subquantizer);

        for _ in 0..IVF_ITERATIONS {
            refine_pq_codebook(&samples, subquantizer, &mut codebook);
        }

        codebooks.extend(codebook);
    }

    codebooks
}

fn sample_residuals(
    references: &impl ReferenceView,
    centroids: &[ReferenceVector],
    assignments_by_position: &[[u32; IVF_ASSIGNMENTS_PER_REFERENCE]],
    requested_count: usize,
) -> Vec<ReferenceVector> {
    let entry_count = references.len() * IVF_ASSIGNMENTS_PER_REFERENCE;
    let sample_count = requested_count.min(entry_count);
    let mut samples = Vec::with_capacity(sample_count);

    for sample in 0..sample_count {
        let entry = sample * entry_count / sample_count;
        let position = entry / IVF_ASSIGNMENTS_PER_REFERENCE;
        let assignment = entry % IVF_ASSIGNMENTS_PER_REFERENCE;
        let vector = references.vector_at(position);
        let centroid = &centroids[assignments_by_position[position][assignment] as usize];
        samples.push(residual_vector(&vector, centroid));
    }

    samples
}

fn initial_pq_codebook(samples: &[ReferenceVector], subquantizer: usize) -> Vec<PqSubVector> {
    let mut codebook = Vec::with_capacity(PQ_CODEWORDS);

    for codeword in 0..PQ_CODEWORDS {
        let index = codeword * samples.len() / PQ_CODEWORDS;
        codebook.push(subvector(&samples[index], subquantizer));
    }

    codebook
}

fn refine_pq_codebook(
    samples: &[ReferenceVector],
    subquantizer: usize,
    codebook: &mut [PqSubVector],
) {
    let mut sums = vec![[0_i64; PQ_DIMENSIONS_PER_SUBQUANTIZER]; PQ_CODEWORDS];
    let mut counts = vec![0_u32; PQ_CODEWORDS];

    for sample in samples {
        let value = subvector(sample, subquantizer);
        let codeword = nearest_codeword(codebook, &value);
        counts[codeword] += 1;

        for dimension in 0..PQ_DIMENSIONS_PER_SUBQUANTIZER {
            sums[codeword][dimension] += i64::from(value[dimension]);
        }
    }

    for codeword in 0..PQ_CODEWORDS {
        let count = counts[codeword];

        if count == 0 {
            continue;
        }

        for dimension in 0..PQ_DIMENSIONS_PER_SUBQUANTIZER {
            codebook[codeword][dimension] = (sums[codeword][dimension] / i64::from(count)) as i16;
        }
    }
}

fn encode_residual(
    vector: &ReferenceVector,
    centroid: &ReferenceVector,
    codebooks: &[PqSubVector],
    output: &mut [u8],
) {
    let residual = residual_vector(vector, centroid);

    for (subquantizer, byte) in output.iter_mut().enumerate() {
        let value = subvector(&residual, subquantizer);
        let start = subquantizer * PQ_CODEWORDS;
        *byte = nearest_codeword(&codebooks[start..start + PQ_CODEWORDS], &value) as u8;
    }
}

fn residual_vector(vector: &ReferenceVector, centroid: &ReferenceVector) -> ReferenceVector {
    let mut residual = [0; VECTOR_DIMENSIONS];

    for dimension in 0..VECTOR_DIMENSIONS {
        residual[dimension] = vector[dimension].saturating_sub(centroid[dimension]);
    }

    residual
}

fn subvector(vector: &ReferenceVector, subquantizer: usize) -> PqSubVector {
    let mut output = [0; PQ_DIMENSIONS_PER_SUBQUANTIZER];
    let start = subquantizer * PQ_DIMENSIONS_PER_SUBQUANTIZER;

    output.copy_from_slice(&vector[start..start + PQ_DIMENSIONS_PER_SUBQUANTIZER]);
    output
}

fn nearest_codeword(codebook: &[PqSubVector], value: &PqSubVector) -> usize {
    let mut nearest = 0;
    let mut nearest_distance = u64::MAX;

    for (index, codeword) in codebook.iter().enumerate() {
        let distance = pq_distance2_limited(value, codeword, nearest_distance);

        if distance < nearest_distance {
            nearest = index;
            nearest_distance = distance;
        }
    }

    nearest
}

fn pq_distance2_limited(a: &PqSubVector, b: &PqSubVector, limit: u64) -> u64 {
    let mut distance = 0;

    for (left, right) in a.iter().zip(b) {
        let delta = i64::from(*left) - i64::from(*right);
        distance += (delta * delta) as u64;

        if distance >= limit {
            return distance;
        }
    }

    distance
}

fn nearest_centroid(centroids: &[ReferenceVector], vector: &ReferenceVector) -> usize {
    let mut nearest = 0;
    let mut nearest_distance = u64::MAX;

    for (index, centroid) in centroids.iter().enumerate() {
        let distance = distance2_limited(vector, centroid, nearest_distance);

        if distance < nearest_distance {
            nearest = index;
            nearest_distance = distance;
        }
    }

    nearest
}

fn nearest_centroids<const N: usize>(
    centroids: &[ReferenceVector],
    vector: &ReferenceVector,
) -> [usize; N] {
    let mut nearest = [0; N];
    let mut nearest_distances = [u64::MAX; N];
    let mut filled = 0;
    let mut farthest_slot = 0;

    for (index, centroid) in centroids.iter().enumerate() {
        let limit = if filled < N {
            u64::MAX
        } else {
            nearest_distances[farthest_slot]
        };
        let distance = distance2_limited(vector, centroid, limit);

        if filled < N {
            nearest[filled] = index;
            nearest_distances[filled] = distance;

            if distance > nearest_distances[farthest_slot] {
                farthest_slot = filled;
            }

            filled += 1;
        } else if distance < nearest_distances[farthest_slot] {
            nearest[farthest_slot] = index;
            nearest_distances[farthest_slot] = distance;
            farthest_slot = farthest_slot_in(&nearest_distances);
        }
    }

    nearest
}

fn farthest_slot_in(distances: &[u64]) -> usize {
    let mut farthest_slot = 0;

    for slot in 1..distances.len() {
        if distances[slot] > distances[farthest_slot] {
            farthest_slot = slot;
        }
    }

    farthest_slot
}

fn distance2_limited(a: &ReferenceVector, b: &ReferenceVector, limit: u64) -> u64 {
    let mut distance = 0;

    for (left, right) in a.iter().zip(b) {
        let delta = i64::from(*left) - i64::from(*right);
        distance += (delta * delta) as u64;

        if distance >= limit {
            return distance;
        }
    }

    distance
}

fn write_index(
    output: &Path,
    reference_count: usize,
    centroids: &[ReferenceVector],
    codebooks: &[PqSubVector],
    offsets: &[u32],
    indices: &[u32],
    codes: &[u8],
) -> Result<()> {
    let output_file = File::create(output)?;
    let mut writer = BufWriter::new(output_file);
    writer.write_all(IVF_MAGIC)?;
    writer.write_all(&(reference_count as u64).to_le_bytes())?;
    writer.write_all(&(centroids.len() as u32).to_le_bytes())?;
    writer.write_all(&(indices.len() as u64).to_le_bytes())?;
    writer.write_all(&(PQ_SUBQUANTIZERS as u32).to_le_bytes())?;
    writer.write_all(&(PQ_CODEWORDS as u32).to_le_bytes())?;
    writer.write_all(&(PQ_DIMENSIONS_PER_SUBQUANTIZER as u32).to_le_bytes())?;

    for centroid in centroids {
        for value in centroid {
            writer.write_all(&value.to_le_bytes())?;
        }
    }

    for codeword in codebooks {
        for value in codeword {
            writer.write_all(&value.to_le_bytes())?;
        }
    }

    for offset in offsets {
        writer.write_all(&offset.to_le_bytes())?;
    }

    for index in indices {
        writer.write_all(&index.to_le_bytes())?;
    }

    writer.write_all(codes)?;
    writer.flush()?;
    Ok(())
}

struct ReferenceDataset {
    mmap: Mmap,
    count: usize,
    fraud_offset: usize,
}

trait ReferenceView: Sync {
    fn len(&self) -> usize;
    fn index_at(&self, position: usize) -> usize;
    fn vector_at(&self, position: usize) -> ReferenceVector;
}

impl ReferenceDataset {
    fn load(path: &Path) -> Result<Self> {
        let input_file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&input_file) }?;

        if mmap.len() < REFERENCE_HEADER_LEN {
            bail!("invalid references binary: file is smaller than header");
        }

        if &mmap[..REFERENCE_MAGIC.len()] != REFERENCE_MAGIC {
            bail!("invalid references binary: bad magic");
        }

        let mut count_bytes = [0; size_of::<u64>()];
        count_bytes.copy_from_slice(&mmap[REFERENCE_MAGIC.len()..REFERENCE_HEADER_LEN]);
        let count = u64::from_le_bytes(count_bytes) as usize;
        let fraud_offset = REFERENCE_HEADER_LEN + count * VECTOR_LEN;
        let expected_len = fraud_offset + count.div_ceil(8);

        if mmap.len() != expected_len {
            bail!(
                "invalid references binary: expected {} bytes, got {} bytes",
                expected_len,
                mmap.len()
            );
        }

        Ok(Self {
            mmap,
            count,
            fraud_offset,
        })
    }

    fn len(&self) -> usize {
        self.count
    }

    fn vector_at(&self, index: usize) -> ReferenceVector {
        let offset = REFERENCE_HEADER_LEN + index * VECTOR_LEN;
        let reference = unsafe { self.mmap.as_ptr().add(offset).cast::<i16>() };
        let mut vector = [0; VECTOR_DIMENSIONS];

        for (dimension, value) in vector.iter_mut().enumerate() {
            *value = i16::from_le(unsafe { *reference.add(dimension) });
        }

        vector
    }

    fn is_fraud(&self, index: usize) -> bool {
        let byte = self.mmap[self.fraud_offset + index / 8];
        byte & (1 << (index % 8)) != 0
    }
}

struct AllReferences<'a> {
    references: &'a ReferenceDataset,
}

impl<'a> AllReferences<'a> {
    fn new(references: &'a ReferenceDataset) -> Self {
        Self { references }
    }
}

impl ReferenceView for AllReferences<'_> {
    fn len(&self) -> usize {
        self.references.len()
    }

    fn index_at(&self, position: usize) -> usize {
        position
    }

    fn vector_at(&self, position: usize) -> ReferenceVector {
        self.references.vector_at(position)
    }
}

struct LabelReferences<'a> {
    references: &'a ReferenceDataset,
    indices: Vec<usize>,
}

impl<'a> LabelReferences<'a> {
    fn new(references: &'a ReferenceDataset, is_fraud: bool) -> Self {
        let mut indices = Vec::new();

        for index in 0..references.len() {
            if references.is_fraud(index) == is_fraud {
                indices.push(index);
            }
        }

        Self {
            references,
            indices,
        }
    }
}

impl ReferenceView for LabelReferences<'_> {
    fn len(&self) -> usize {
        self.indices.len()
    }

    fn index_at(&self, position: usize) -> usize {
        self.indices[position]
    }

    fn vector_at(&self, position: usize) -> ReferenceVector {
        self.references.vector_at(self.index_at(position))
    }
}
