# Flow

- O índice é lido no startup a partir de `resources/hnsw.bin`.
- Cada request é vetorizada em `i16[14]`.
- A busca usa HNSW com distância L2 exata.
- Níveis superiores fazem caminhada gulosa a partir do entrypoint.
- O nível 0 faz busca com beam fixo.
- O `fraud_score` é calculado como fraudes no top 5 dividido por 5.
- Casos com mistura de fraude e legítimo no top 5 são marcados como boundary.
- A resposta HTTP é escolhida de uma tabela JSON estática a partir do `fraud_score`.
