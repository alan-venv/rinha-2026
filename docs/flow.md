## Fluxo Runtime

1. A request é desserializada.
2. A transação é normalizada em vetor quantizado de `14` dimensões.
3. O serviço tenta resolver a decisão por busca Morton rápida.
4. Casos não decididos pela busca rápida são tratados como boundary.
5. A resposta final é serializada em JSON.

## Índice

1. As referências são carregadas.
2. Os vetores são convertidos para entradas Morton.
3. As entradas são ordenadas e gravadas em formato binário.
