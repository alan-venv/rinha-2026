## Fluxo Runtime

- O índice é lido no startup a partir de `resources/ivf.bin`.
- O arquivo do índice é acessado via `memmap2`; o runtime não copia o índice inteiro para heap.
- O índice atual usa IVF direto com `4096` centroids.

1. A request é desserializada.
2. A transação é normalizada em vetor quantizado de `16` dimensões.
3. A query é comparada contra todos os `4096` centroids.
4. Os `8` centroids mais próximos são usados no primeiro lote.
5. As listas invertidas desses centroids são percorridas.
6. Cada referência candidata é comparada por distância euclidiana ao quadrado.
7. O top `5` vizinhos mais próximos é mantido durante a busca.
8. O `fraud_score` é calculado como fraudes no top 5 dividido por 5.
9. Se o top 5 tiver classes misturadas, a transação é marcada como `boundary_case`.
10. Para `boundary_case`, a busca repete o mesmo fluxo usando `24` centroids.
11. A resposta é escolhida de uma tabela JSON estática a partir do `fraud_score`.

## Diagnóstico

O `diagnose` mede apenas o caminho principal:

- total de entradas
- tempo total de processamento
- quantidade e percentual de `boundary_cases`
- divergências de decisão
- divergências dentro de `boundary_cases`
- divergências fora de `boundary_cases`
