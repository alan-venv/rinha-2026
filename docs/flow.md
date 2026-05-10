## Fluxo Runtime

- O índice é lido no startup a partir de `resources/ivf.bin`.
- O arquivo do índice é acessado via `memmap2`; o runtime não copia o índice inteiro para heap.
- O índice atual usa IVF direto com `4096` centroids.
- Centroids, índices de candidatos e vetores de referência são armazenados em layout flat/AoS.

1. A request é desserializada.
2. A transação é normalizada em vetor quantizado de `16` dimensões.
3. A query é comparada contra todos os `4096` centroids com SSE2 e early discard após `8` dimensões.
4. Até `8` centroids mais próximos são mantidos internamente.
5. O caminho principal consulta apenas os `2` centroids mais próximos.
6. As listas invertidas desses centroids são percorridas integralmente.
7. Cada candidato referencia um vetor flat, lido contiguamente e comparado com SSE2.
8. O top `5` vizinhos mais próximos é mantido durante a busca.
9. O `fraud_score` é calculado como fraudes no top 5 dividido por 5.
10. Se o top 5 tiver classes misturadas, a transação é marcada como `boundary_case`.
11. A resposta é escolhida de uma tabela JSON estática a partir do `fraud_score`.
