use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::thread;

use anyhow::{Result, bail};
use rinha::*;

use crate::reference::ReferenceDataset;
use crate::structs::{AssignmentChunk, Assignments};

pub struct IndexIvf;

impl IndexIvf {
    pub fn build(dataset: &ReferenceDataset) -> Result<()> {
        Self::validate_config(dataset)?;

        let workers = thread::available_parallelism().map_or(1, usize::from);

        println!(
            "building IVF_FLAT: references={}, centroids={}, sample={}, iterations={}, probes={}, workers={}",
            dataset.len(),
            IVF_FINE_CENTROIDS,
            IVF_FINE_SAMPLES,
            IVF_FINE_ITERATIONS,
            IVF_FINE_PROBES,
            workers,
        );

        let centroids = Self::train_centroids(
            dataset,
            IVF_FINE_CENTROIDS,
            IVF_FINE_SAMPLES,
            IVF_FINE_ITERATIONS,
            workers,
        );
        let assignments = Self::assign_references(dataset, &centroids, workers);
        let indices = Self::candidate_indices(&assignments.offsets, assignments.by_position);
        Self::write_index(
            Path::new("resources/ivf.bin"),
            dataset,
            &centroids,
            &assignments.offsets,
            &indices,
        )?;

        println!(
            "wrote {}: {} bytes",
            "resources/ivf.bin",
            IVF_HEADER_LEN
                + centroids.len() * VECTOR_LEN
                + assignments.offsets.len() * size_of::<u32>()
                + indices.len() * size_of::<u32>()
                + dataset.len() * VECTOR_LEN
                + dataset.fraud_bits().len()
        );

        Ok(())
    }

    fn validate_config(dataset: &ReferenceDataset) -> Result<()> {
        if IVF_FINE_CENTROIDS > dataset.len() {
            bail!(
                "invalid centroid count: {} > {}",
                IVF_FINE_CENTROIDS,
                dataset.len()
            );
        }

        if IVF_FINE_PROBES == 0 || IVF_FINE_PROBES > IVF_FINE_CENTROIDS {
            bail!(
                "invalid probe count: {} for {} centroids",
                IVF_FINE_PROBES,
                IVF_FINE_CENTROIDS
            );
        }

        Ok(())
    }

    fn train_centroids(
        dataset: &ReferenceDataset,
        centroid_count: usize,
        sample_count: usize,
        iterations: usize,
        workers: usize,
    ) -> Vec<ReferenceVector> {
        let samples = Self::sample_references(dataset, sample_count.max(centroid_count));
        let mut centroids = Self::initial_centroids(&samples, centroid_count);
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
                            let centroid = Self::nearest_centroid(centroids, sample);
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
        dataset: &ReferenceDataset,
        requested_count: usize,
    ) -> Vec<ReferenceVector> {
        let sample_count = requested_count.min(dataset.len());
        let mut samples = Vec::with_capacity(sample_count);

        for sample in 0..sample_count {
            let index = sample * dataset.len() / sample_count;
            samples.push(dataset.vector_at(index));
        }

        samples
    }

    fn initial_centroids(
        samples: &[ReferenceVector],
        centroid_count: usize,
    ) -> Vec<ReferenceVector> {
        let mut centroids = Vec::with_capacity(centroid_count);

        for centroid in 0..centroid_count {
            let index = centroid * samples.len() / centroid_count;
            centroids.push(samples[index]);
        }

        centroids
    }

    fn assign_references(
        dataset: &ReferenceDataset,
        centroids: &[ReferenceVector],
        workers: usize,
    ) -> Assignments {
        let workers = workers.min(dataset.len().max(1));
        let chunk_len = dataset.len().div_ceil(workers);
        let chunks = thread::scope(|scope| {
            let mut tasks = Vec::with_capacity(workers);

            for chunk_start in (0..dataset.len()).step_by(chunk_len) {
                let chunk_end = (chunk_start + chunk_len).min(dataset.len());
                tasks.push(scope.spawn(move || {
                    let mut counts = vec![0_u32; centroids.len()];
                    let mut assignments = Vec::with_capacity(chunk_end - chunk_start);

                    for position in chunk_start..chunk_end {
                        let vector = dataset.vector_at(position);
                        let centroid = Self::nearest_centroid(centroids, &vector);
                        assignments.push(centroid as u32);
                        counts[centroid] += 1;
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

        let mut by_position = Vec::with_capacity(dataset.len());
        let mut centroid_counts = vec![0_u32; centroids.len()];

        for chunk in ordered_chunks {
            println!(
                "assigned {} references",
                (chunk.start + chunk.assignments.len()).min(dataset.len())
            );

            by_position.extend(chunk.assignments);

            for centroid in 0..centroids.len() {
                centroid_counts[centroid] += chunk.centroid_counts[centroid];
            }
        }

        let offsets = Self::candidate_list_offsets(&centroid_counts);

        Assignments {
            offsets,
            by_position,
        }
    }

    fn candidate_list_offsets(centroid_counts: &[u32]) -> Vec<u32> {
        let mut offsets = vec![0_u32; centroid_counts.len() + 1];

        for centroid in 0..centroid_counts.len() {
            offsets[centroid + 1] = offsets[centroid] + centroid_counts[centroid];
        }

        offsets
    }

    fn candidate_indices(offsets: &[u32], by_position: Vec<u32>) -> Vec<u32> {
        let mut cursors = offsets[..offsets.len() - 1].to_vec();
        let mut indices = vec![0_u32; by_position.len()];

        for (position, centroid) in by_position.into_iter().enumerate() {
            let cursor = &mut cursors[centroid as usize];
            indices[*cursor as usize] = position as u32;
            *cursor += 1;
        }

        indices
    }

    fn nearest_centroid(centroids: &[ReferenceVector], vector: &ReferenceVector) -> usize {
        let mut nearest = 0;
        let mut nearest_distance = u64::MAX;

        for (index, centroid) in centroids.iter().enumerate() {
            let distance = Self::distance2_limited(vector, centroid, nearest_distance);

            if distance < nearest_distance {
                nearest = index;
                nearest_distance = distance;
            }
        }

        nearest
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
        dataset: &ReferenceDataset,
        centroids: &[ReferenceVector],
        offsets: &[u32],
        indices: &[u32],
    ) -> Result<()> {
        let output_file = File::create(output)?;
        let mut writer = BufWriter::new(output_file);
        writer.write_all(IVF_MAGIC)?;
        writer.write_all(&(dataset.len() as u64).to_le_bytes())?;
        writer.write_all(&(centroids.len() as u32).to_le_bytes())?;
        writer.write_all(&(indices.len() as u64).to_le_bytes())?;

        for centroid in centroids {
            Self::write_vector(&mut writer, centroid)?;
        }

        for offset in offsets {
            writer.write_all(&offset.to_le_bytes())?;
        }

        for index in indices {
            writer.write_all(&index.to_le_bytes())?;
        }

        for vector in &dataset.vectors {
            Self::write_vector(&mut writer, vector)?;
        }

        writer.write_all(dataset.fraud_bits())?;
        writer.flush()?;
        Ok(())
    }

    fn write_vector(writer: &mut BufWriter<File>, vector: &ReferenceVector) -> Result<()> {
        for value in vector {
            writer.write_all(&value.to_le_bytes())?;
        }

        Ok(())
    }
}
