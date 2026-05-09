## Fluxo Runtime

- O índice `resources/ivf.bin` é embutido no binário via `include_bytes!` e copiado para heap no startup; o runtime não lê arquivo de índice.

1. A request é desserializada.
2. A transação é normalizada em vetor quantizado de 16 dimensões.
3. A query é comparada contra `1024` coarse centroids.
4. Os `32` coarse centroids mais próximos são guardados no contexto de busca.
5. A busca usa os `15` melhores coarse groups para selecionar candidatos fine.
6. A query é comparada contra os fine centroids desses grupos.
7. Os `12` fine centroids mais próximos são selecionados.
8. As listas invertidas desses fine centroids são percorridas.
9. Cada referência candidata é comparada por distância euclidiana ao quadrado.
10. O top `5` vizinhos mais próximos é mantido durante a busca.
11. O `fraud_score` é calculado como fraudes no top 5 dividido por 5.
12. A resposta é escolhida de uma tabela JSON estática.
