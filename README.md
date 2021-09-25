# Previsão de Cancelamento! Olist Store
![download](https://user-images.githubusercontent.com/85971408/134776393-773c03f7-06d4-4665-90d4-6ee6563dc14b.png)

# Como prever quais clientes vão cancelar suas compras em um “e-commerce”?

---

## O problema:

Após a aprovação do pagamento, a logística e o vendedor precisam contrair custos para entregar o pedido aos seus consumidores, no entanto, por vezes os mesmos consumidores cancelam seus pedidos pelos mais diversos motivos, além do cancelamento, onerar o “e-commerce” com a perda de parte do faturamento, onera também toda a estrutura logística, que se vê obrigada a trazer o produto de volta ao vendedor.

## Como resolver isso com Data Science?

Criando um algorítimo que retorne um ranking de clientes mais propensos a cancelar! Tendo acesso a esse ranking, a plataforma pode acionar a equipe de produto para garantir que o cliente permaneça fiel à compra realizada, assim evitando prejuízos.

**Nesse “Read-me” você encontra um resumo de cada passo efetuado no projeto, além de claro poder observar os resultados finais graficamente & financeiramente. Espero que goste!**

### Passo 1, Carregar e "Tipar" os dados:

Resumo das operações feitas na fase de tipagem, onde os tipos das variáveis são corrigidos.

Ao total, temos 29 colunas no dataframe, maior parte dessas veio com a tipagem correta, no entanto, algumas colunas precisaram de correção:

- Convertendo datas de “string” para Datas:

```python
df['DATA_LIMITE_ENTREGA_PARCEIRO_LOGISTICO'] = pd.to_datetime(df['DATA_LIMITE_ENTREGA_PARCEIRO_LOGISTICO'], format = '%Y-%m-%d %H:%M:%S' , errors = 'coerce')

df['DATA_PAGAMENTO'] = pd.to_datetime(df['DATA_PAGAMENTO'], format = '%Y-%m-%d %H:%M:%S' , errors = 'coerce')

df['DATA_APROVACAO_PAGAMENTO'] = pd.to_datetime(df['DATA_APROVACAO_PAGAMENTO'], format = '%Y-%m-%d %H:%M:%S' , errors = 'coerce')

df['DATA_POSTAGEM'] = pd.to_datetime(df['DATA_POSTAGEM'], format = '%Y-%m-%d %H:%M:%S' , errors = 'coerce')

df['DATA_ESTIMADA_ENTREGA'] = pd.to_datetime(df['DATA_ESTIMADA_ENTREGA'], format = '%Y-%m-%d %H:%M:%S' , errors = 'coerce')

df['DATA_ENTREGUE'] = pd.to_datetime(df['DATA_ENTREGUE'], format = '%Y-%m-%d %H:%M:%S' , errors = 'coerce')
```

- Convertendo CEP's para string:

```python
df['PREFIXO_CEP_CLIENTE'] = df['PREFIXO_CEP_CLIENTE'].apply(str)

df['PREFIXO_CEP_VENDEDOR'] = df['PREFIXO_CEP_VENDEDOR'].apply(str)
```

### Passo 2, Limpeza de nulos:

Resumo do passo de limpeza geral, onde valores nulos são substituídos de diversas maneiras.

```python
df.isnull().sum()
```

```
ID_ORDEM                                     0
ID_CLIENTE                                   0
ID_PRODUTO                                   0
ID_VENDEDOR                                  0
DATA_LIMITE_ENTREGA_PARCEIRO_LOGISTICO       0
DATA_PAGAMENTO                               0
DATA_APROVACAO_PAGAMENTO                    15
DATA_POSTAGEM                             1245
DATA_ESTIMADA_ENTREGA                        0
DATA_ENTREGUE                             2567
QUANTIA_ITEMS_NESSA_ORDEM                    0
QUANTIA_METODOS_PAGAMENTO                    0
METODO_PAGAMENTO                             0
QUANTIA_PARCELAS                             0
PRECO_SEM_FRETE                              0
PRECO_FRETE                                  0
VALOR_COMPRA                                 0
NOME_CATEGORIA_PRODUTO                    1698
QUANTIA_FOTOS_ANUNCIO                     1698
PESO_EM_GRAMAS                              20
COMPRIMENTO_EM_CENTIMETROS                  20
LARGURA_PRODUTO_EM_CENTIMETROS              20
ALTURA_PRODUTO_EM_CENTIMETROS               20
PREFIXO_CEP_CLIENTE                          0
CIDADE_CLIENTE                               0
ESTADO_CLIENTE                               0
PREFIXO_CEP_VENDEDOR                         0
CIDADE_VENDEDOR                              0
ESTADO_VENDEDOR                              0
TARGET_STATUS_DA_ORDEM                       0
```

Algumas colunas vieram com uma quantia de dados nulos, efetuei um processo diferente de preenchimento em cada uma:

1. Data do Pagamento:

    Essa coluna tinha apenas 15 registros nuloes, e em todos, o método de pagamento foi o boleto, provalvelmente, a entrega do pedido ocorreu antes da devida compensação do boleto, como são apenas 15 registros, e nenhum da classe "cancelado" (sendo meu alvo mais importante), optei por retira-los do dataset.

    ```python
    df = df[df['DATA_APROVACAO_PAGAMENTO'].notna()]
    ```

2. Data de Postagem:

    Temos 1254 registros com essa data nula, no entanto, a maior dos pedidos cancelados tem essa coluna nula justamente por terem sidos cancelados, entenda-se: “a data de postagem é nula, pois o cliente cancelou o pedido antes da postagem ser feita”. Portanto, essa coluna não pode ser usada no modelo.

    Tendo em vista que isso seria um vazamento de dados futuro.

3. Nome da Categoria do produto:

    Ao todo 1698 produtos não possuem categorias. Optei por dar à esses produtos uma característica própria de "produtos sem categoria"

    ```python
    df['NOME_CATEGORIA_PRODUTO'].loc[df['NOME_CATEGORIA_PRODUTO'] == 'sem_categoria'] # feito
    ```

4. Quantia de fotos no Anúncio:

    Como esperado, muitos dos produtos sem categoria também não possuem a quantia de fotos registradas (provavelmente por não terem foto no site) aqui minha abordagem foi de preencher os campos de quantia nula com 0.

    ```python
    df['QUANTIA_FOTOS_ANUNCIO'].loc[df['QUANTIA_FOTOS_ANUNCIO'] == 0] # feito
    ```

5. Medidas:

    Apenas 20 produtos não possuem nenhum registro de peso, altura, comprimento, ou largura, e a grande maioria desses produtos pertence a categoria anteriormente criada de "produtos sem categoria":

    ```python
    df['NOME_CATEGORIA_PRODUTO'].loc[df['PESO_EM_GRAMAS'].isnull()]
    ```

    ```
    8650      sem_categoria
    9138      sem_categoria
    9139      sem_categoria
    22269     sem_categoria
    22270     sem_categoria
    22271     sem_categoria
    33345     sem_categoria
    37621     sem_categoria
    42352     sem_categoria
    42353     sem_categoria
    45384             bebes
    48753     sem_categoria
    54070     sem_categoria
    65001     sem_categoria
    80337     sem_categoria
    82712     sem_categoria
    85859     sem_categoria
    86409     sem_categoria
    87936     sem_categoria
    105429    sem_categoria
    ```

    Para fazer o preenchimento, foi calculado a média dos dados dessa categoria:

    ```python
    df.loc[df['NOME_CATEGORIA_PRODUTO'] =='sem_categoria'].mean()
    ```

    ```
    QUANTIA_ITEMS_NESSA_ORDEM            1.168533
    QUANTIA_METODOS_PAGAMENTO            1.142015
    QUANTIA_PARCELAS                     2.535651
    PRECO_SEM_FRETE                    112.496523
    PRECO_FRETE                         18.033677
    VALOR_COMPRA                       148.931167
    QUANTIA_FOTOS_ANUNCIO                0.000000
    PESO_EM_GRAMAS                    1881.230036
    COMPRIMENTO_EM_CENTIMETROS          26.594756
    LARGURA_PRODUTO_EM_CENTIMETROS      20.178784
    ALTURA_PRODUTO_EM_CENTIMETROS       14.696067
    PREFIXO_CEP_CLIENTE                       inf
    PREFIXO_CEP_VENDEDOR                      inf
    dtype: float64
    ```

    Preenchendo com as médias:

    ```python
    df['COMPRIMENTO_EM_CENTIMETROS'].fillna(26.59, inplace=True)
    df['PESO_EM_GRAMAS'].fillna(1881.23, inplace=True)
    df['LARGURA_PRODUTO_EM_CENTIMETROS'].fillna(20.17, inplace=True)
    df['ALTURA_PRODUTO_EM_CENTIMETROS'].fillna(14.69, inplace=True)
    ```

    ### Passo 3, Mapa de Hipóteses:

    Resumo da fase de criação de hipóteses para serem validadas nos próximos passos.

    Através do site [https://coggle.it](https://coggle.it/) foi feito um "brainstorm" onde imaginei diferentes agentes para o fato: "Cancelamento de Compra" além de imaginar os agentes, também elenquei suas caraterísticas, para formular hipóteses de oque poderia causar mais cancelamentos:

   ![Cancelamento_de_Compra](https://user-images.githubusercontent.com/85971408/134775966-03be32b3-1f6b-4101-831c-aac923adf709.png)


    Com base nesse “brainstorm” foram postuladas diferentes hipóteses, por exemplo:

    - Quanto mais caro o frete mais cancelam?
    - Quanto menos produtos anteriormente comprados maior o cancelamento?
    - Quanto menos “conhecido” o vendedor, mais cancelamentos?
    - Quanto mais próximo do final do mês, cancelam mais?

Obs: por se tratar de um resumo não elenquei todas, o processo pode ser visto em integralidade no notebook.

No próximo passo, as variáveis necessárias para verificar todas as hipóteses criadas foram derivadas dos dados.

### Passo 4, Criação de variáveis:

Resumo da fase de criação de variáveis, que servirão para o passo de análise exploratória e também aos próximos, onde o modelo pode utilizar as variáveis criadas.

A maioria das hipóteses pode ser verificada apenas com as variáveis já disponíveis. No entanto, algumas precisaram ser criadas.

* Tempo até aprovarem o pagamento!

```python
# subtraindo a data do pagamento da data da aprovação
df1['TEMPO_APROVACAO'] = df1['DATA_APROVACAO_PAGAMENTO'] - df1['DATA_PAGAMENTO']

# convertendo para time delta
df1['TEMPO_APROVACAO'] = pd.to_timedelta(df1['TEMPO_APROVACAO'])
```

* Quantia de compras anteriores:

```python
# ordenando por data
df1 = df1.sort_values(by='DATA_PAGAMENTO')

# contando aparições do id
df1['COMPRAS_TOTAIS_ID'] = df1.groupby('ID_CLIENTE')['ID_CLIENTE'].cumcount() + 1
```

* Popularidade do vendedor:

```python
df1['POPULARIDADE_VENDEDOR'] = df1.groupby('ID_VENDEDOR')['ID_VENDEDOR'].cumcount() + 1
```

* Vendedor inter-estadual:

```python
df1['DISTANTE'] = np.where(df1['ESTADO_VENDEDOR'] == df1['ESTADO_CLIENTE'], 0, 1)
```

* Popularidade da categoria do produto:

```python
df1['POPULARIDADE_CATEGORIA'] = df1.groupby('NOME_CATEGORIA_PRODUTO').cumcount() +1
```

* Tempo Desde a última compra:

```python
# ordenando por id do cliente & data

df1 = df1.sort_values(['ID_CLIENTE', 'DATA_PAGAMENTO'])

df1['TEMPO_DESDE_ULTIMO_PEDIDO'] = 'sem_pedido_anterior'

for rep in range(len(df1)):
    if df1['ID_CLIENTE'].iloc[rep] == df1['ID_CLIENTE'].iloc[rep - 1]:
        df1['TEMPO_DESDE_ULTIMO_PEDIDO'].iloc[rep] = df1['DATA_PAGAMENTO'].iloc[rep] - df1['DATA_PAGAMENTO'].iloc[rep-1]
```

7, 8, 9, 10, 11. Compras feitas em diferentes dimensões de tempo:

```python
# dia da semana
df1['DIA_DA_SEMANA'] = df1['DATA_PAGAMENTO'].dt.dayofweek

# dia do mes
df1['DIA_DO_MES'] = df1['DATA_PAGAMENTO'].dt.day

# mes
df1['MES'] = df1['DATA_PAGAMENTO'].dt.month

# semana do ano
df1['SEMANA_ANO'] = df1['DATA_PAGAMENTO'].dt.weekofyear
```

1. Tempo de previsão:

```python
df1['PREVISAO_DEMORA'] = df1['DATA_ESTIMADA_ENTREGA'] - df1['DATA_PAGAMENTO']

# convertendo para time delta
df1['PREVISAO_DEMORA'] = pd.to_timedelta(df1['PREVISAO_DEMORA'])
```

### Passo 5, Análise Exploratória de Dados.

Resumo da fase de análise exploratória, onde foram validadas (ou não) as hipóteses criadas anteriormente.

Obs: Por se tratar de um resumo apresentarei somente os gráficos mais importantes, a análise completa pode ser encontrada no notebook do projeto!

### Passo 5.1, Análise Univariada:

Variáveis observadas sozinhas, histogramas e distribuições.

- Variável Resposta:

    ```python
    # change fig-size
    sns.set(rc={'figure.figsize':(12, 6)})

    # plot
    sns.countplot(data=df, 
                  x='TARGET_STATUS_DA_ORDEM',
                 palette='rocket');
    ```

  ![download_(2)](https://user-images.githubusercontent.com/85971408/134775998-7af8a425-4019-413e-b694-edd78b723e55.png)


    Variável resposta extremamente desequilibrada, necessário atenção para treinar os modelos.

### Passo 5.2, Análise Bivariada:

Análise de como as variáveis se relacionam com a variável resposta, observando a validade ou não das hipóteses:

---

- Alguns métodos de pagamento podem gerar mais cancelamentos percentualmente?

    ```python
    # df apenas clientes cancelados
    df_cancelados = df.loc[df['TARGET_STATUS_DA_ORDEM'] == 'canceled']

    # plot
    sns.countplot(data=df_cancelados,
                  x='METODO_PAGAMENTO',
                 palette='rocket');
    ```

![download_(1)](https://user-images.githubusercontent.com/85971408/134776004-1b523598-0c6c-4eaf-9cd7-ee19e40d283d.png)


O cartão de crédito foi mais usado que outros meios, no entanto, como isso se compara com o uso de diferentes métodos em outras categorias que não são canceladas?

```python
x, y ='TARGET_STATUS_DA_ORDEM', 'METODO_PAGAMENTO'
dfaux = df.groupby(x)[y].value_counts(normalize=True).mul(100)
dfaux
```

```
approved                credit_card         100.000000
canceled                credit_card          76.855124
                        boleto               16.961131
                        voucher               5.123675
                        debit_card            1.060071
delivered               credit_card          73.810414
                        boleto               19.428094
                        voucher               5.323468
                        debit_card            1.438023
invoiced                credit_card          70.080863
                        boleto               24.528302
                        voucher               3.773585
                        debit_card            1.617251
processing              credit_card          67.466667
                        boleto               25.866667
                        voucher               6.133333
                        debit_card            0.533333
```

Cartão de crédito utilizado 3,04% a mais que na classe majoritária, diferença percentual de ambas sai do boleto, bem mais comum entre pagadores que não cancelam, representação gráfica:

```python
# plot

sns.barplot(data=dfaux.loc[(dfaux['TARGET_STATUS_DA_ORDEM'] == 'canceled')
                           | (dfaux['TARGET_STATUS_DA_ORDEM'] == 'delivered')], 
            x='TARGET_STATUS_DA_ORDEM', 
            y='percentual',
           hue='METODO_PAGAMENTO',
           palette='rocket');
```

![download_(5)](https://user-images.githubusercontent.com/85971408/134776013-d7837899-c7f9-4998-9661-893e6b189a23.png)


---

- Preço, produtos mais caros são mais cancelados?

```python
df[['VALOR_COMPRA', 'TARGET_STATUS_DA_ORDEM']].groupby('TARGET_STATUS_DA_ORDEM').mean().reset_index()
```

![Untitled](https://user-images.githubusercontent.com/85971408/134776028-47367f8d-8f24-4b03-919c-f6480aa51f85.png)


A média de preço dos produtos cancelados é a maior, dando a entender que sim, no entanto, ordens ainda em processamento também mostram uma média de compras grande, portanto as ordens canceladas tem um maior preço aparente, mas que pode ser coincidência, vamos observar a diferença nas distribuições de ambas:

```python
# tirar outliers, melhor vis

dfaux1 = df.loc[df['VALOR_COMPRA'] < 700.00]

plt.subplot(1,2,1)
sns.boxplot(data=dfaux1, 
           x='TARGET_STATUS_DA_ORDEM',
           y='VALOR_COMPRA',
           order=['canceled', 'delivered'],
           palette='rocket');

plt.subplot(1,2,2)
sns.histplot(data=(df.loc[(df['TARGET_STATUS_DA_ORDEM'] == 'delivered') 
                          & (df['VALOR_COMPRA'] < 700.00)
                          | (df['TARGET_STATUS_DA_ORDEM'] == 'canceled') 
                          & (df['VALOR_COMPRA'] < 700.00)]),
            hue='TARGET_STATUS_DA_ORDEM',
            x='VALOR_COMPRA',
            log_scale=(False, True),
            multiple="layer", 
            kde=True,
            palette ='rocket');
```

![download_(6)](https://user-images.githubusercontent.com/85971408/134776033-e839ee4d-2e00-4b52-998c-0663ba31e2dd.png)


Parece ter uma “pausa” no declínio da distribuição entre 400 e 500 R$ na classe “canceled”

Há uma leve propenção as compras da classe cancelada serem mais caras.

---

- Pagamentos que demoram mais para serem aprovados geram mais cancelamentos?

```python
df[['TEMPO_APROVACAO', 'TARGET_STATUS_DA_ORDEM']].groupby('TARGET_STATUS_DA_ORDEM').mean(numeric_only=False).reset_index()
```

![Untitled 1](https://user-images.githubusercontent.com/85971408/134776073-8fb8275c-f30f-414f-8f7d-5260990c36a7.png)


Ordens canceladas apresentaram um tempo de espera superior em 2 horas comparadas a classe majoritária “delivered”.

```python
sns.barplot(data=df_agrupado,
           x='TARGET_STATUS_DA_ORDEM',
           y=(df_agrupado['TEMPO_APROVACAO']).values.astype(np.int64),
           palette='rocket');
```

![download_(7)](https://user-images.githubusercontent.com/85971408/134776058-5833589a-a1b5-4e95-b80e-7650a0fd1a0e.png)


Observe a diferença entre a classe cancelada e a classe majoritária, “delivered”.

---

- Algumas cidades geram mais cancelamentos percentuais?

```python
 # retornar um df com os cancelamentos por cidade.
dfaux1 = df[['CIDADE_CLIENTE', 'TARGET_STATUS_DA_ORDEM']].loc[df['TARGET_STATUS_DA_ORDEM'] == 'canceled'].groupby('CIDADE_CLIENTE').count().sort_values('TARGET_STATUS_DA_ORDEM', ascending=False).reset_index()

# retornar pedidos gerais por cidade, incluso cancelamentos.
dfaux2 = df[['CIDADE_CLIENTE', 'TARGET_STATUS_DA_ORDEM']].groupby('CIDADE_CLIENTE').count().sort_values('TARGET_STATUS_DA_ORDEM', ascending=False).reset_index()

# cria uma coluna que mostre o percentual de cancelamentos por cidade
dfaux2['CANCELAMENTO_PERCENTUAL'] = (dfaux1['TARGET_STATUS_DA_ORDEM']*100) / dfaux2['TARGET_STATUS_DA_ORDEM']
dfaux2['CANCELAMENTO_PERCENTUAL'] = dfaux2['CANCELAMENTO_PERCENTUAL'].fillna(0)
dfaux2['CANCELAMENTO_PERCENTUAL'] = dfaux2['CANCELAMENTO_PERCENTUAL'].astype(float)
dfaux2 = dfaux2.sort_values('CANCELAMENTO_PERCENTUAL', ascending=False)

dfaux2.sort_values('CANCELAMENTO_PERCENTUAL', ascending=False)
```

![Untitled 2](https://user-images.githubusercontent.com/85971408/134776164-9eed818e-5b24-4e48-a868-cce9c819450d.png)


Ao que aparenta sim, algumas cidades geram mais cancelamentos, porém também precisamos considerar a população da cidade, pois cidades pequenas podem ter um descolamento do percentual de cancelamento com pouco pedidos.

```python
sns.set(rc={'figure.figsize':(10, 50)})

sns.barplot(data=dfaux2,
           y=dfaux2['CIDADE_CLIENTE'].loc[dfaux2['CANCELAMENTO_PERCENTUAL'] > 0.01],
           x=dfaux2['CANCELAMENTO_PERCENTUAL'].loc[dfaux2['CANCELAMENTO_PERCENTUAL'] > 0.01],
           palette='rocket');
```

![download_(9)](https://user-images.githubusercontent.com/85971408/134776167-5b2531bb-44f7-4dcd-ab5c-ce62e9b78049.png)


Observando a correlação entre o cancelamento percentual e o número de pedidos da cidade:

```python
dfaux3 = dfaux2.loc[dfaux2['CANCELAMENTO_PERCENTUAL'] != 0]

sns.lmplot(data=dfaux3,
         y='CANCELAMENTO_PERCENTUAL',
         x='TARGET_STATUS_DA_ORDEM',
         logistic=True,
         aspect=1*3);
```

![download_(10)](https://user-images.githubusercontent.com/85971408/134776169-176dd98c-3393-4fe7-ab9a-68154703ba08.png)


Existe uma correlação negativa entre a quantia de cancelamentos e a quantia de pedidos, oque mostra que muitas vezes o cancelamento percentual pode ser alto em algumas cidades pequenas devido ao fácil aumento do mesmo com poucos pedidos cancelados, no entanto, há um “ponto” fora dessa relação, onde temos a maior concentração de pedidos e um cancelamento percentual não tão baixo.

```python
buble = px.scatter(dfaux3,
                  x='QUANTIA_PEDIDOS',
                  y='CANCELAMENTO_PERCENTUAL',
                  size='QUANTIA_PEDIDOS',
                  color='CIDADE_CLIENTE',
                  size_max=30)

buble.show()
```

![Untitled 3](https://user-images.githubusercontent.com/85971408/134776183-a1bca567-3e8b-4757-89b3-590c3eed1ac4.png)

Apesar de num geral os cancelamentos diminuírem com cidades com mais pedidos, o São Paulo é um outlier á regra, tendo o maior número de pedidos e mesmo assim um grande cancelamento percentual.

---

- Vendedores impopulares tendem a gerar mais cancelamentos?

    Para analisar isso, criaremos grupos de vendedores, e classificá-los como “populares”, “impopulares” ou “médios”.

    ```python
    df['POPULARIDADE_VENDEDOR'].describe()
    ```

    ```
    count    117585.000000
    mean        223.066820
    std         364.113765 
    min           1.000000
    25%          19.000000 -- Impopular!
    50%          71.000000 -- Médio!
    75%         236.000000 -- Popular!
    max        2133.000000
    ```

```python
dfaux = df.copy()

# crianco os buckets
dfaux['GRUPO_POPULARIDADE'] = 0
dfaux.loc[df['POPULARIDADE_VENDEDOR'] < 19, 'GRUPO_POPULARIDADE'] = 'BAIXA'
dfaux.loc[(df['POPULARIDADE_VENDEDOR'] <= 71) & (df['POPULARIDADE_VENDEDOR'] >= 19), 'GRUPO_POPULARIDADE'] = 'MÉDIA'
dfaux.loc[(df['POPULARIDADE_VENDEDOR'] > 71) & (df['POPULARIDADE_VENDEDOR'] <= 236), 'GRUPO_POPULARIDADE'] = 'ALTA'
dfaux.loc[df['POPULARIDADE_VENDEDOR'] > 236, 'GRUPO_POPULARIDADE'] = 'MUITO_ALTA'
```

```python
sns.set(rc={'figure.figsize':(12, 5)})

# distribuição das popularidades geral

sns.countplot(data=dfaux,
             x='GRUPO_POPULARIDADE',
             palette='rocket');
```

![download_(11)](https://user-images.githubusercontent.com/85971408/134776188-7599c76e-98f0-4701-9226-41b57e94fa71.png)


```python
# df cancelados
dfaux1 = dfaux.loc[dfaux['TARGET_STATUS_DA_ORDEM'] == 'canceled']

# observando distribuições na classe cancelada
sns.countplot(data=dfaux1,
             x='GRUPO_POPULARIDADE',
             palette='rocket');
```

![download_(12)](https://user-images.githubusercontent.com/85971408/134776197-61da1135-c9d5-4749-b4b7-0e9fe3c6e076.png)


Vendedores impopulares são bem mais frequentes entre os pedidos cancelados.

Considerando que no passo de criação de fatures, fiz a popularidade dos vendedores crescer conforme o número de aparições ordenado pelo tempo, é importante observar qual a mediana do tempo entre pedidos cancelados e pedidos gerais, para entender se isso não pode estar enviesando a “popularidade” dos vendedores na classe cancelada.

```python
dfaux['DATA_PAGAMENTO'] = pd.to_datetime(dfaux['DATA_PAGAMENTO'])
dfaux1['DATA_PAGAMENTO'] = pd.to_datetime(dfaux1['DATA_PAGAMENTO'])

# normais
dfaux['DATA_PAGAMENTO'].mean()
# Timestamp('2017-12-30 17:19:30.144099840')

# cancelados
dfaux1['DATA_PAGAMENTO'].mean()
# Timestamp('2017-11-30 02:26:17.971727104')
```

Para uma última conferida, vamos plotar 3 Gráficos, um com as popularidades gerais, outros com apenas os 566 exemplos das popularidades da classe cancelada, e outro com 566 exemplos aleatórios:

```python
plt.subplot(1,3,1)

sns.countplot(data=dfaux,                 # geral
             x='GRUPO_POPULARIDADE',
             palette='rocket');

plt.subplot(1,3,2)
sns.countplot(data=dfaux.sample(566),     # aleatórios
             x='GRUPO_POPULARIDADE',
             palette='rocket');

plt.subplot(1,3,3)
sns.countplot(data=dfaux1,                # cancelado em cinza
             x='GRUPO_POPULARIDADE',
             palette='Greys_r');
```

![download_(15)](https://user-images.githubusercontent.com/85971408/134776209-d326a7fb-74c5-4a1c-beb4-7ab37c51c71d.png)


Existe uma clara propensão às compras canceladas estarem atreladas à vendedores impopulares.

---

- Vendedores que estão mais longe do comprador geram mais cancelamentos?

Lembrando:

0 — Estado do vendedor igual ao do comprador.

1 — Estado do comprador diferente do estado do vendedor.

```python
# plotando 3 gráficos: geral, cancelados, amostragem com 566 dados aleatórios

plt.subplot(1,3,1)

sns.countplot(data=df.loc[df['TARGET_STATUS_DA_ORDEM'] != 'canceled'],
         x='DISTANTE',
            palette='gist_gray');

plt.subplot(1,3,2)

sns.countplot(data=df.sample(566),
         x='DISTANTE',
            palette='gist_gray');

plt.subplot(1,3,3)                # df com os cancelados em laranja

sns.countplot(data=df.loc[df['TARGET_STATUS_DA_ORDEM'] == 'canceled'],
         x='DISTANTE',
            palette='OrRd_r');
```

![download_(16)](https://user-images.githubusercontent.com/85971408/134776220-b8c81dc3-7532-4cc1-b4da-5e587c973dd3.png)


Sim, é estranho, mas por alguma razão os compradores que compram produtos do mesmo estado que eles cancelam mais percentualmente, tendo em vista que no dataset geral a maiorias das compras efetuadas é inter estadual, mas não na classe cancelada.

---

- Compras com mais itens normalmente sofrem mais cancelamentos?

```python
df[['TARGET_STATUS_DA_ORDEM', 'QUANTIA_ITEMS_NESSA_ORDEM']].groupby('TARGET_STATUS_DA_ORDEM').mean().reset_index()
```

![Untitled 4](https://user-images.githubusercontent.com/85971408/134776230-f6b4b6ff-672f-4c92-9e09-940a69c0a889.png)


A classe "canceled" tem uma quantia ligeiramente maior de produtos por ordem.

Abaixo irie plotar um histograma com 566 exemplos entregues vs. 566 pedidos cancelados, para observar se existe uma diferença na densidade das distribuições de quantias de pedidos por ordem entre ambas as classes.

```python
f, ax = plt.subplots(figsize=(14, 7))

sns.histplot(data=(df.loc[(df['TARGET_STATUS_DA_ORDEM'].sample(566) == 'delivered') 
                          & (df['QUANTIA_ITEMS_NESSA_ORDEM'] <= 8)
                          | (df['TARGET_STATUS_DA_ORDEM'] == 'canceled') 
                          & (df['QUANTIA_ITEMS_NESSA_ORDEM'] <= 8)]),
            hue='TARGET_STATUS_DA_ORDEM',
            x='QUANTIA_ITEMS_NESSA_ORDEM',
            log_scale=(False, True), # facilitar visualização
            multiple="layer", 
            kde=True,
            bins=6,
            palette ='rocket');

ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8]); # até 8 items por ordem
```

![download_(17)](https://user-images.githubusercontent.com/85971408/134776240-e103578d-70a5-44ee-a985-39d6da6b6f4c.png)


A classe de pedidos cancelados tem uma leve predominância em sua densidade em compras com 4 & 5 pedidos.

---

- Quantia de fotos:

```python
df[['TARGET_STATUS_DA_ORDEM', 'QUANTIA_FOTOS_ANUNCIO']].groupby('TARGET_STATUS_DA_ORDEM').mean()
```

![Untitled 5](https://user-images.githubusercontent.com/85971408/134776242-c974b35c-ab93-4f43-ac03-d7a8d9caa165.png)


A classe de pedidos cancelada aparenta ter um número bem menor de fotos na média.

```python
hist = px.histogram(df.loc[(df['TARGET_STATUS_DA_ORDEM'] == 'delivered') | (df['TARGET_STATUS_DA_ORDEM'] == 'canceled')],
                   x='QUANTIA_FOTOS_ANUNCIO',
                   y='QUANTIA_FOTOS_ANUNCIO',
                   marginal='box',
                   log_y=True,
                   color='TARGET_STATUS_DA_ORDEM')
hist.show()
```

![Untitled 6](https://user-images.githubusercontent.com/85971408/134776246-12d92745-37ed-41fb-ad96-2bf2449eb267.png)


A distribuição das fotos mostra que entre pedidos cancelados a quantia de fotos realmente é mais baixa.

---

- Algumas categorias de produtos geram mais cancelamentos?

```python
sns.set(rc={'figure.figsize':(14, 50)})

df_cancelados = df.loc[df['TARGET_STATUS_DA_ORDEM'] =='canceled']

plt.subplot(1,2,1)
sns.countplot(data=df_cancelados, # cancelados
             y='NOME_CATEGORIA_PRODUTO',
             palette='rocket',
             order=df['NOME_CATEGORIA_PRODUTO'].value_counts().index);

plt.subplot(1,2,2)
sns.countplot(data=df, # geral
             y='NOME_CATEGORIA_PRODUTO',
             palette='Dark2',
             order=df_cancelados['NOME_CATEGORIA_PRODUTO'].value_counts().index);
```

![Untitled 7](https://user-images.githubusercontent.com/85971408/134776262-fde4ca4b-4afd-40a1-8d46-888412f8f1df.png)


Sim, ordenando o histograma da classe cancelada e comparando com o geral é possível ver que algumas categorias são mais canceladas no “df_cancelado” que no dataframe original, com destaque para a categoria de “cama_mesa_banho”, a mais comprada, mas pouco cancelada.

---

- Popularidade da categoria, assim como a popularidade do vendedor, observaremos se categorias menos populares geram mais cancelamentos, novamente, a popularidade é calculada com base no número de aparições daquela categoria no tempo, portanto, ire usar 2 dataframes para comparar a popularidade das categorias:
    - Data frame com a classe “canceled”
    - Dataframe com a classe geral, mas com data mínima é máxima iguais ao dataframe da classe “canceled”.

    ```python
    # definindo datas mínimas e máximas

    x = df_cancelados['DATA_PAGAMENTO'].max()
    y = df_cancelados['DATA_PAGAMENTO'].min()

    # separando partição do df entre as datas x e y
    dfaux1 = df.loc[(df['TARGET_STATUS_DA_ORDEM'] != 'canceled') 
    							& (df['DATA_PAGAMENTO'] < x) 
    							& (df['DATA_PAGAMENTO'] > y)]

    # criando um df com os 566 exemplos cancelados + 566 exemplos no mesmo espaço de tempo
    dfaux1 = dfaux1.sample(566)
    dfaux1 = dfaux1.append(df_cancelados)

    # plot
    violin = px.violin(dfaux1.loc[(dfaux1['TARGET_STATUS_DA_ORDEM'] == 'delivered') 
    															| (dfaux1['TARGET_STATUS_DA_ORDEM'] == 'canceled')],
                      y='POPULARIDADE_CATEGORIA',
                      color='TARGET_STATUS_DA_ORDEM',
                      box=True)

    violin.show()
    ```

   ![Untitled 8](https://user-images.githubusercontent.com/85971408/134776270-e9030d23-8019-4d85-885a-eb3c9e6fabb6.png)


    A classe “canceled” tem uma mediana menor além de uma concentração maior de classes com popularidades um pouco mais baixas.

    ```python
    sns.set(rc={'figure.figsize':(15,10)})

    sns.histplot(data=dfaux1.loc[(dfaux1['TARGET_STATUS_DA_ORDEM'] == 'delivered') | (dfaux1['TARGET_STATUS_DA_ORDEM'] == 'canceled')],
                x='POPULARIDADE_CATEGORIA',
                hue='TARGET_STATUS_DA_ORDEM',
                palette='rocket',
                 element="step",
                kde=True);
    ```
    
    ![Untitled 9](https://user-images.githubusercontent.com/85971408/134776273-f5a66feb-0c30-4e7d-85ac-2e67d561bdf1.png)


    Densidade da classe cancelada um pouco maior em produtos de categoria com popularidade mais baixa.

    ---

- Compras feitas no “fim de mês” geram mais cancelamentos?

```python
df[['DIA_DO_MES', 'TARGET_STATUS_DA_ORDEM']].groupby('TARGET_STATUS_DA_ORDEM').mean().reset_index()
```

![Untitled 10](https://user-images.githubusercontent.com/85971408/134776284-3c0493b7-c94d-48c5-b8be-90879048e183.png)

A média de dia do mês é ligeiramente maior na classe cancelados, vamos explorar melhor.

```python
sns.set(rc={'figure.figsize':(10.7,9.27)})

sns.boxenplot(data=df.loc[(df['TARGET_STATUS_DA_ORDEM'] == 'delivered') | (df['TARGET_STATUS_DA_ORDEM'] == 'canceled')],
             y='DIA_DO_MES',
             palette='rocket',
             x='TARGET_STATUS_DA_ORDEM');
```

![Untitled 11](https://user-images.githubusercontent.com/85971408/134776292-5aa27e9e-f24b-4366-a1cc-d651d3270983.png)


Parece haver uma diferença muito pequena entre as caixas do “boxenplot” dos pedidos cancelados, com caixas indicando uma densidade menor nos primeiros dias do mês e ligeiramente maior no final do mês.

 Vamos para uma última exploração com noavmente 566 exemplos gerais vs. os 566 exemplos de cancelamento.

```python
dfaux1 = df.sample(566).loc[df['TARGET_STATUS_DA_ORDEM'] != 'canceled']
dfaux1 = dfaux1.append(df_cancelados)

sns.histplot(data=dfaux1.loc[(dfaux1['TARGET_STATUS_DA_ORDEM'] == 'delivered') 
                         | (dfaux1['TARGET_STATUS_DA_ORDEM'] == 'canceled')],
             x='DIA_DO_MES',
             palette='rocket',
             hue='TARGET_STATUS_DA_ORDEM',
             stat="density",
             cumulative=True,
             fill=False,
             common_norm=False,
             element="step",
            bins=30);
```

![Untitled 12](https://user-images.githubusercontent.com/85971408/134776300-9d5e644c-3c79-4de5-81e6-918eafc84f78.png)


Sei que histogramas cumulativos são um pouco difíceis de entender, mas é possível notar uma leve diferença na velocidade da acumulação entre uma barra e outra, com a classe de pedidos cancelados tendo menos acúmulo de densidade após o dia 15, e com o acúmulo voltando mais próximo à regra geral ao final do mês.

Plotagem de um histograma um pouco mais claro:

```python
dfaux1 = df.loc[df['TARGET_STATUS_DA_ORDEM'] != 'canceled']
dfaux1 = dfaux1.append(df_cancelados)

sns.histplot(data=dfaux1.loc[(dfaux1['TARGET_STATUS_DA_ORDEM'] == 'delivered') 
                         | (dfaux1['TARGET_STATUS_DA_ORDEM'] == 'canceled')],
             x='DIA_DO_MES',
             palette='rocket',
             hue='TARGET_STATUS_DA_ORDEM',
             stat="density",
              common_norm=False,
             kde=True,
            bins=30);
```

![Untitled 13](https://user-images.githubusercontent.com/85971408/134776301-0f4db3c7-d79e-4801-baf4-289415e56cdc.png)

Leve diminuição da densidade entre os dias 15 & 20 nos pedidos cancelados.

---

- Mês do ano, alguns meses geram mais cancelamentos?

```python
df[['MES', 'TARGET_STATUS_DA_ORDEM']].groupby('TARGET_STATUS_DA_ORDEM').mean().reset_index()
```

![Untitled 14](https://user-images.githubusercontent.com/85971408/134776305-1eff7e5c-6f65-4de8-ab6a-70413e77c9e2.png)


Pedidos cancelados tendem a ser comprados antes da classe majoritárias de pedidos que são os entregues.

```python
plt.subplot(1, 3, 1)

sns.histplot(data=df_cancelados, # cancelados
             x='MES',
             stat='density',
             bins=12);

plt.subplot(1, 3, 2)
																 # geral
sns.histplot(data=df.loc[df['TARGET_STATUS_DA_ORDEM'] != 'canceled'],
             x='MES',
             stat='density',
             bins=12);

plt.subplot(1, 3, 3)
																# exemplos aleatórios
sns.histplot(data= df.sample(566).loc[dfaux1['TARGET_STATUS_DA_ORDEM'] != 'canceled'],
             x='MES',
             stat='density',
             bins=12);
```

![Untitled 15](https://user-images.githubusercontent.com/85971408/134776309-fb3673d8-b8ec-44c3-97da-762a903788de.png)


Entre 566 exemplos aleatórios e o dataframe todo a diferença dos meses mais comuns de compra é pequena, sendo os meses com mais e menos vendas quase sempre os mesmos, já nos dados de compras cancelados temos diferenças entre os meses mais e menos comuns:

- Fevereiro representa 17,6% das vendas canceladas, contra menos de 9% compras gerais.
- Junho representa 11% das vendas gerais, mas apenas 5,1% das canceladas.
- Dezembro representa 6% das vendas gerais, mas apenas 1,25% das vendas canceladas.

---

### Passo 5.3, Análise Multivariada:

Resumo da parte onde vejo correlações entre variáveis, procurando por variáveis a serem retiradas devido à alta correlação.

```python
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

dfnum = df.select_dtypes(include=numerics)

correlation = dfnum.corr(method='pearson')
sns.heatmap(correlation, annot=True);
```

![Untitled 16](https://user-images.githubusercontent.com/85971408/134776316-ff648179-868a-43fa-8066-eeb56ac396f0.png)


Há uma correlação alta entre as variáveis que resumem o tamanho do produto, portanto simplificarei a altura, comprimento, & largura para volume.

Também há uma correlação com a quantia de meios de pagamento que um comprador usa naquela compra e o número total de compras anteriores que eles realizou, apesar de ser estranho verifiquei as causas para isso, mas não cheguei em nenhuma conclusão diferente de coincidência.

Vendo correlação entre as categóricas com matriz de confusão:

```python
# cramer v
def cramer_v(x, y):
    
    cm = pd.crosstab(x, y).to_numpy()
    
    n = cm.sum()

    r, k = cm.shape

    chi2 = ss.chi2_contingency(cm)[0]
    chi2corr = max(0, chi2 - (k-1)*(r-1)/(n-1))
    kcorr = k - (k-1)**2/(n-1)
    rcorr = r - (r-1)**2/(n-1)

    v = np.sqrt((chi2corr/n) / (min(kcorr-1, rcorr-1)))
    return v

dfcat = df.select_dtypes('object')

# dropando categóricas que não me interessam na matriz de confusão, ouq ue não irão para o modelo
dfcat = dfcat.drop('ID_ORDEM', axis=1)
dfcat = dfcat.drop('ID_CLIENTE', axis=1)
dfcat = dfcat.drop('ID_VENDEDOR', axis=1)
dfcat = dfcat.drop('GRUPO_POPULARIDADE', axis=1)
dfcat = dfcat.drop('TEMPO_DESDE_ULTIMO_PEDIDO', axis=1)
dfcat = dfcat.drop('ID_PRODUTO', axis=1)

dfcat = df.select_dtypes('object')

# cramer V

a1 = cramer_v( dfcat['state_holiday'], dfcat['state_holiday'] )
a2 = cramer_v( dfcat['state_holiday'], dfcat['store_type'] )
a3 = cramer_v( dfcat['state_holiday'], dfcat['assortment'] )

a4 = cramer_v( dfcat['store_type'], dfcat['state_holiday'] )
a5 = cramer_v( dfcat['store_type'], dfcat['store_type'] )
a6 = cramer_v( dfcat['sore_type'], dfcat['assortment'] )

a7 = cramer_v( a['assortment'], a['state_holiday'] )
a8 = cramer_v( a['assortment'], a['store_type'] )
a9 = cramer_v( a['assortment'], a['assortment'] )

d = pd.DataFrame( {'state_holiday': [a1, a2, a3],
'store_type': [a4, a5, a6],
'assortment': [a7, a8, a9] })

d = d.set_index( d.columns )
sns.heatmap( d, annot=True )

contador = 0
lista = []
ç = pd.DataFrame()

for i in dfcat.columns:   
    
    for n in dfcat.columns:
        
        a = cramer_v(dfcat[i], dfcat[n])
        lista.append(a)
        a = None
        
    dfx = pd.DataFrame({i: lista})
    ç[i] = dfx[i]
    lista = []

ç = ç.set_index(ç.columns)

sns.heatmap(data=ç, annot=True);
```

![Untitled 17](https://user-images.githubusercontent.com/85971408/134776321-da7d4248-c2b7-4471-b5a8-9a92ffc13d5c.png)


Nenhuma correlação muito alta fora a Cidade e o Estado (por motivos óbvios), ficarei apenas com a Cidade nos modelos de ML.

Retirando as variáveis & simplificando o dataset para classe majoritária & cancelada.

```python
df = df[df['TARGET_STATUS_DA_ORDEM'].isin(['delivered', 'canceled'])]

drop_cols = ['ID_ORDEM', 'ID_CLIENTE', 'COMPRIMENTO_EM_CENTIMETROS',
            'LARGURA_PRODUTO_EM_CENTIMETROS', 'ALTURA_PRODUTO_EM_CENTIMETROS', 'ESTADO_CLIENTE', 'ESTADO_VENDEDOR']

# agregando as medidas do produto que tinham muita relação como VOLUME
df['DIMENSAO'] = df['LARGURA_PRODUTO_EM_CENTIMETROS']*df['ALTURA_PRODUTO_EM_CENTIMETROS']*df['COMPRIMENTO_EM_CENTIMETROS']

df.drop(drop_cols, axis=1, inplace=True)
```

### Passo 6, Modelagem, Scalling, & Encoding

Resumo do passo onde as variáveis são transformadas para facilitar o acerto dos modelos de machine learning.

Como muitas variáveis de dimensionalidade de tempo foram importantes na EDA, criarei ciclidade ao tempo, mais detalhes podem ser encontrados aqui:

[https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/](https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/)

[https://www.kaggle.com/avanwyk/encoding-cyclical-features-for-deep-learning](https://www.kaggle.com/avanwyk/encoding-cyclical-features-for-deep-learning)

```python
# day of week
df['DIA_DA_SEMANA_sin'] = df['DIA_DA_SEMANA'].apply(lambda x: np.sin(x *(2*np.pi/7)))
df['DIA_DA_SEMANA_cos'] = df['DIA_DA_SEMANA'].apply(lambda x: np.cos(x *(2*np.pi/7)))

# month
df['MES_sin'] = df['MES'].apply(lambda x: np.sin(x *(2*np.pi/12)))
df['MES_cos'] = df['MES'].apply(lambda x: np.cos(x *(2*np.pi/12)))

# day
df['DIA_DO_MES_sin'] = df['DIA_DO_MES'].apply(lambda x: np.sin(x *(2*np.pi/30)))
df['DIA_DO_MES_cos'] = df['DIA_DO_MES'].apply(lambda x: np.cos(x *(2*np.pi/30)))

# week of year
df['SEMANA_ANO_sin'] = df['SEMANA_ANO'].apply(lambda x: np.sin(x *(2*np.pi/52)))
df['SEMANA_ANO_cos'] = df['SEMANA_ANO'].apply(lambda x: np.cos(x *(2*np.pi/52)))
```

Colocando as variáveis de duração no mesmo padrão, minutos:

```python
# definindo pessoas sem pedido anterior como se tivessem pedido há 10 anos

df['TEMPO_DESDE_ULTIMO_PEDIDO'].loc[df['TEMPO_DESDE_ULTIMO_PEDIDO'].str.contains('sem_pedido_anterior')] = '3650 days 00:00:00'

# convertendo tipos
df['TEMPO_APROVACAO'] = pd.to_timedelta(df['TEMPO_APROVACAO'])
df['PREVISAO_DEMORA'] = pd.to_timedelta(df['PREVISAO_DEMORA'])
df['TEMPO_DESDE_ULTIMO_PEDIDO'] = pd.to_timedelta(df['TEMPO_DESDE_ULTIMO_PEDIDO'])

# função e correção
def days_hours_minutes(td):
    x = td.days, td.seconds//3600, (td.seconds//60)%60
    x1 = x[0] * 1440
    x2 = x[1] * 60
    x3 = x[2]
    
    td = x1 + x2 + x3
    return td

contador = 0

for rep in range(len(df)):
                 
    df['TEMPO_APROVACAO'].iloc[contador] = days_hours_minutes(df['TEMPO_APROVACAO'].iloc[contador])
    df['PREVISAO_DEMORA'].iloc[contador] = days_hours_minutes(df['PREVISAO_DEMORA'].iloc[contador])
    df['TEMPO_DESDE_ULTIMO_PEDIDO'].iloc[contador] = days_hours_minutes(df['TEMPO_DESDE_ULTIMO_PEDIDO'].iloc[contador])
    
    contador += 1 
    

# convertendo inteiro
df['TEMPO_APROVACAO'] = df['TEMPO_APROVACAO'].astype(int)
df['PREVISAO_DEMORA'] = df['PREVISAO_DEMORA'].astype(int)
df['TEMPO_DESDE_ULTIMO_PEDIDO'] = df['TEMPO_DESDE_ULTIMO_PEDIDO'].astype(int)
```

Escalonando as variáveis numéricas, com RobustScaler em variáveis com muitos outliers, e MinMaxScaler em variáveis mais “Gaussianas”.

```python

# separando colunas que podem ser utilizadas
y = df['TARGET_STATUS_DA_ORDEM']
y = np.where(y == 'delivered', 0, 1) # convertendo classes para 0 e 1
df_sem_y = df.drop('TARGET_STATUS_DA_ORDEM', axis=1)

x_treino, x_teste, y_treino, y_teste = train_test_split(df_sem_y, y, test_size= 0.33, random_state=32)

r = RobustScaler()

x_treino['QUANTIA_ITEMS_NESSA_ORDEM'] = r.fit_transform(x_treino[['QUANTIA_ITEMS_NESSA_ORDEM']].values)
x_treino['QUANTIA_METODOS_PAGAMENTO'] = r.fit_transform(x_treino[['QUANTIA_METODOS_PAGAMENTO']].values)
x_treino['QUANTIA_PARCELAS'] = r.fit_transform(x_treino[['QUANTIA_PARCELAS']].values)
x_treino['PRECO_FRETE'] = r.fit_transform(x_treino[['PRECO_FRETE']].values)
x_treino['VALOR_COMPRA'] = r.fit_transform(x_treino[['VALOR_COMPRA']].values)
x_treino['QUANTIA_FOTOS_ANUNCIO'] = r.fit_transform(x_treino[['QUANTIA_FOTOS_ANUNCIO']].values)
x_treino['PESO_EM_GRAMAS'] = r.fit_transform(x_treino[['PESO_EM_GRAMAS']].values)
x_treino['COMPRAS_TOTAIS_ID'] = r.fit_transform(x_treino[['COMPRAS_TOTAIS_ID']].values)
x_treino['POPULARIDADE_VENDEDOR'] = r.fit_transform(x_treino[['POPULARIDADE_VENDEDOR']].values)
x_treino['POPULARIDADE_CATEGORIA'] = r.fit_transform(x_treino[['POPULARIDADE_CATEGORIA']].values)
x_treino['TEMPO_DESDE_ULTIMO_PEDIDO'] = r.fit_transform(x_treino[['TEMPO_DESDE_ULTIMO_PEDIDO']].values)
x_treino['TEMPO_APROVACAO'] = r.fit_transform(x_treino[['TEMPO_APROVACAO']].values)
x_treino['DIMENSAO'] = r.fit_transform(x_treino[['DIMENSAO']].values)
x_treino['PREVISAO_DEMORA'] = r.fit_transform(x_treino[['PREVISAO_DEMORA']].values)

x_teste['QUANTIA_ITEMS_NESSA_ORDEM'] = r.fit_transform(x_teste[['QUANTIA_ITEMS_NESSA_ORDEM']].values)
x_teste['QUANTIA_METODOS_PAGAMENTO'] = r.fit_transform(x_teste[['QUANTIA_METODOS_PAGAMENTO']].values)
x_teste['QUANTIA_PARCELAS'] = r.fit_transform(x_teste[['QUANTIA_PARCELAS']].values)
x_teste['PRECO_FRETE'] = r.fit_transform(x_teste[['PRECO_FRETE']].values)
x_teste['VALOR_COMPRA'] = r.fit_transform(x_teste[['VALOR_COMPRA']].values)
x_teste['QUANTIA_FOTOS_ANUNCIO'] = r.fit_transform(x_teste[['QUANTIA_FOTOS_ANUNCIO']].values)
x_teste['PESO_EM_GRAMAS'] = r.fit_transform(x_teste[['PESO_EM_GRAMAS']].values)
x_teste['COMPRAS_TOTAIS_ID'] = r.fit_transform(x_teste[['COMPRAS_TOTAIS_ID']].values)
x_teste['POPULARIDADE_VENDEDOR'] = r.fit_transform(x_teste[['POPULARIDADE_VENDEDOR']].values)
x_teste['POPULARIDADE_CATEGORIA'] = r.fit_transform(x_teste[['POPULARIDADE_CATEGORIA']].values)
x_teste['TEMPO_DESDE_ULTIMO_PEDIDO'] = r.fit_transform(x_teste[['TEMPO_DESDE_ULTIMO_PEDIDO']].values)
x_teste['TEMPO_APROVACAO'] = r.fit_transform(x_teste[['TEMPO_APROVACAO']].values)
x_teste['DIMENSAO'] = r.fit_transform(x_teste[['DIMENSAO']].values)
x_teste['PREVISAO_DEMORA'] = r.fit_transform(x_teste[['PREVISAO_DEMORA']].values)
         
MinMax = MinMaxScaler()

x_treino['MES'] = MinMax.fit_transform(x_treino[['MES']].values)
x_treino['DIA_DA_SEMANA'] = MinMax.fit_transform(x_treino[['DIA_DA_SEMANA']].values)
x_treino['DIA_DO_MES'] = MinMax.fit_transform(x_treino[['DIA_DO_MES']].values)
x_treino['SEMANA_ANO'] = MinMax.fit_transform(x_treino[['SEMANA_ANO']].values)

x_teste['MES'] = MinMax.fit_transform(x_teste[['MES']].values)
x_teste['DIA_DA_SEMANA'] = MinMax.fit_transform(x_teste[['DIA_DA_SEMANA']].values)
x_teste['DIA_DO_MES'] = MinMax.fit_transform(x_teste[['DIA_DO_MES']].values)
x_teste['SEMANA_ANO'] = MinMax.fit_transform(x_teste[['SEMANA_ANO']].values)
```

Encoding em variáveis categóricas:

```python
cat_cols = x_treino.select_dtypes('O').columns.tolist()

encoder = OneHotEncoder(top_categories=2, variables=cat_cols, drop_last=False)
encoder.fit(x_treino)

pkl.dump(encoder, open('preparation/transformation_cat_cols.pkl', 'wb'))

# transform the data
x_treino = encoder.transform(x_treino)
x_teste = encoder.transform(x_teste)
```

### Passo 7, Seleção de features

Resumo do passo de seleção de features.

Nesse passo foi utilizado um algorítimo chamado Boruta, conhecido por selecionar features mais importantes de maneira robusta. Mais informações pode ser encontradas aqui: 

[https://towardsdatascience.com/boruta-explained-the-way-i-wish-someone-explained-it-to-me-4489d70e154a](https://towardsdatascience.com/boruta-explained-the-way-i-wish-someone-explained-it-to-me-4489d70e154a)

Devido ao dataset ser desbalanceado, duas versões do Boruta foram rodadas:

1. Boruta com x_treino, e y_treino equilibrados.
2. Boruta com x_treino, e y_treino sem equilibrar.

Exemplo do Boruta equilibrado:

```python
# random forest
rf = RandomForestClassifier()

# colocando x_treino no formato correto e equilibrando classes
nr = NearMiss()

x_treino_nr_array = x_treino_nr.values
y_treino_nr_vetor = y_treino_nr

#instanciando e rodando o Boruta com 1000 arvorês
boruta = BorutaPy(rf, n_estimators=1000, verbose=False).fit(x_treino_array, y_treino_vetor);

# melhores features
cols_selected = boruta.support_.tolist()
cols_selected_boruta_equilib = x_treino.iloc[:, cols_selected]

# mostrar colunas selecionadas
cols_selected_boruta_equilib.columns

'''

Index(['QUANTIA_PARCELAS', 'PRECO_FRETE', 'VALOR_COMPRA',
       'QUANTIA_FOTOS_ANUNCIO', 'PESO_EM_GRAMAS', 'TEMPO_APROVACAO',
       'COMPRAS_TOTAIS_ID', 'POPULARIDADE_VENDEDOR', 'POPULARIDADE_CATEGORIA',
       'TEMPO_DESDE_ULTIMO_PEDIDO', 'DIA_DO_MES', 'MES', 'SEMANA_ANO',
       'PREVISAO_DEMORA', 'DIMENSAO', 'DIA_DO_MES_sin', 'SEMANA_ANO_sin',
       'SEMANA_ANO_cos', 'METODO_PAGAMENTO_credit_card',
       'METODO_PAGAMENTO_boleto'],
      dtype='object')

'''
```

**Entre as 20 colunas selecionadas pelo Boruta, 15 são colunas criadas na fase de “feature enginering”, oque além de ser ótimo para minha autoestima, mostra como a etapa de criar hipóteses e derivar features é importante para o projeto!**

### Passo 8, Testes com modelos:

Resumo dos melhores modelos testados e validados.

Tivemos duas opiniões a respeito da importância das colunas pelo Boruta, na versão desequilibrada do dataset ele julga 20 colunas como importantes, já na versão equilibrada do teste ele seleciona apenas 2.

Portanto, é necessário:

1. Treinar os modelos com 20 colunas e dataset equilibrado.
2. Treinar os modelos com 20 colunas e dataset desequilibrado.
3. Treinar os modelos com 2 colunas e dataset equilibrado.
4. Treinar os modelos com 2 colunas e dataset desequilibrado.

A métrica escolhida sera a AUC, e os modelos irão dizer a probabilidade um cliente ser da classe “canceled” ou “delivered” 0, e “canceled” 1.

Melhores modelos:

- Random Forest:

    **A melhor AUC foi de 0.791**, com o dataframe desequilibrado, e todas as colunas.

    **Outro destaque foi de AUC 0.781**, com o dataframe desequilibrado, e colunas selecionadas pelo Boruta equilibrado.

    ```python
    x_treino_rf_desequilibrado = x_treino[cols_selected_boruta_equilib.columns]

    x_teste_rf_desequilibrado = x_teste[cols_selected_boruta_equilib.columns]

    y_treino_rf_desequilibrado = y_treino

    # random forest
    rf = RandomForestClassifier(n_estimators=1000, random_state=32)

    # fit
    rf.fit(x_treino_rf_desequilibrado, y_treino_rf_desequilibrado)

    # fazendo previsoes
    previsoes_rf = rf.predict_proba(x_teste_rf_desequilibrado)[:,1]

    auc(y_teste, previsoes_rf)

    # 0.7817747204465377
    ```

- XGBoost:

    **A melhor AUC foi de 0.774**, com o dataframe desequilibrado, e todas as colunas.

    **Outro destaque foi de AUC 0.768**, com o dataframe desequilibrado, e colunas selecionadas pelo Boruta equilibrado.

    ```python
    x_treino_xg = x_treino[cols_selected_boruta_equilib.columns]

    x_teste_xg = x_teste[cols_selected_boruta_equilib.columns]

    xg = XGBClassifier(n_estimators=1000)

    # fit
    xg.fit(x_treino_xg, y_treino)

    # fazendo previsoes
    previsoes_xg_boost = xg.predict_proba(x_teste_xg)[:, 1]

    auc(y_teste, previsoes_xg_boost)

    # 0.7680883993457398
    ```

### Passo 9, Validação Cruzada & Tunning:

Resumo do teste feito com ambos os modelos, Random Forest & XGBoost, além do teste, ambos os modelos foram tunados.

Aqui realizei a validação cruzada da performance de ambos os modelos, a Random Forest acabou se destacando.

```python
# carregando treino e teste

with open('olist_treino_teste.pkl', 'rb') as f:
    x_treino, y_treino, x_teste, y_teste = pkl.load(f)
    
    
# chamando colunas do boruta equilibrado (melhor resultado na Random Forest no passo 11)

x_treino = x_treino[['QUANTIA_PARCELAS', 'PRECO_FRETE', 'VALOR_COMPRA',
        'QUANTIA_FOTOS_ANUNCIO', 'PESO_EM_GRAMAS', 'TEMPO_APROVACAO',
        'COMPRAS_TOTAIS_ID', 'POPULARIDADE_VENDEDOR', 'POPULARIDADE_CATEGORIA',
        'TEMPO_DESDE_ULTIMO_PEDIDO', 'DIA_DO_MES', 'MES', 'SEMANA_ANO',
        'PREVISAO_DEMORA', 'DIMENSAO', 'DIA_DO_MES_sin', 'SEMANA_ANO_sin',
        'SEMANA_ANO_cos', 'METODO_PAGAMENTO_credit_card',
        'METODO_PAGAMENTO_boleto']]

x_teste = x_teste[['QUANTIA_PARCELAS', 'PRECO_FRETE', 'VALOR_COMPRA',
        'QUANTIA_FOTOS_ANUNCIO', 'PESO_EM_GRAMAS', 'TEMPO_APROVACAO',
        'COMPRAS_TOTAIS_ID', 'POPULARIDADE_VENDEDOR', 'POPULARIDADE_CATEGORIA',
        'TEMPO_DESDE_ULTIMO_PEDIDO', 'DIA_DO_MES', 'MES', 'SEMANA_ANO',
        'PREVISAO_DEMORA', 'DIMENSAO', 'DIA_DO_MES_sin', 'SEMANA_ANO_sin',
        'SEMANA_ANO_cos', 'METODO_PAGAMENTO_credit_card',
        'METODO_PAGAMENTO_boleto']]

# juntando treino e teste

x = np.r_[x_treino, x_teste]
y = np.r_[y_treino, y_teste]

# instanciando Random Forest

rf = RandomForestClassifier(n_estimators=1000)

xg = XGBClassifier(n_estimators=1000)

# cross valid:

RandomForest = []
XGBoost = []

for rep in range(10):
    
    # kfold stratfied (pega dados na mesma proporção de classes do dataset geral)
    
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=rep) 
    
    
    # apendando a média de resultados com 10 splits diferentes em cada loop
    
    # auc com predict proba!
    
    score_rf = cross_val_score(rf, x, y, scoring='roc_auc', cv=kf) 
    score_xg = cross_val_score(xg, x, y, scoring='roc_auc', cv=kf)
    
    RandomForest.append(score_rf.mean())
    XGBoost.append(score_xg.mean())
    
    print(f'Score RF:{score_rf.mean()}')
    print(f'Score XG:{score_xg.mean()}')
    
    print(f'{rep}/9')

Resultados = {'RandomForest': RandomForest,
             'XGBoost': XGBoost}

Resultados = pd.DataFrame.from_dict(Resultados)

Resultados.sort_values(['XGBoost', 'RandomForest'], ascending=False)
```

![Untitled 18](https://user-images.githubusercontent.com/85971408/134776336-d1bbd802-2062-4045-bf3c-2d61882f12e3.png)


Após a validação dos acertos de ambos os modelos, foi feito o tunning de ambos, o modelo destaque foi a Random Forest novamente.

Tunning Random Forest

```python
grid_rf = {
       'n_estimators': [100, 500, 1000, 2000],
       'min_samples_leaf': [1, 5, 10, 40],
       'max_depth':[None, 6, 10, 20],
       'criterion':['gini', 'entropy'],
       'max_features':['auto', 'sqrt', 'log2'] #paper sobre random forests e problemas de classificação
       }

# busca parametros
            
clf = RandomizedSearchCV(rf, 
                        param_distributions=grid_rf, 
                        n_iter=10,
                        verbose=3,
                        scoring='roc_auc',
                        cv = 10 # por padrão ele vai usar StratifiedKfold
                         )
busca_rf = clf.fit(x, y)
busca_rf.best_params_
```

Melhores parâmetros: 

**{'n_estimators': 1000,
 'min_samples_leaf': 1,
 'max_features': 'log2',
 'max_depth': 20,
 'criterion': 'entropy'}**

Testando resultado da Random Forest Tunada com validação cruzada:

```python
rf = RandomForestClassifier(n_estimators = 1000,
       min_samples_leaf = 1,
       max_depth = 20,
       criterion = 'entropy',
       max_features = 'log2')

# cross valid:
RandomForest = []

for rep in range(10):
    
    # kfold stratfied (pega dados na mesma proporção de classes do dataset geral)
    
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=rep) 
    
    
    # apendando a média de resultados com 10 splits diferentes em cada loop
    
    # auc com predict proba!
    
    score_rf = cross_val_score(rf, x, y, scoring='roc_auc', cv=kf)

		RandomForest.append(score_rf.mean())
		print(f'Score RF:{score_rf.mean()}')
		print(f'{rep}/9')
```

![Untitled 19](https://user-images.githubusercontent.com/85971408/134776345-698f9af0-2464-490f-924f-561554106853.png)


**Auc Médio Tunado: 0.846**

**Auc Médio Sem Tunning : 0.824**

Plotando resultados com e sem tunning de ambos os modelos:

![Untitled 20](https://user-images.githubusercontent.com/85971408/134776348-964f46f0-7567-496a-b11e-3c43053e4295.png)


É possível Notar que tanto a RandomForest quanto o XGBoost apresentaram melhora geral de resultados, com a Random Forest sendo melhor em ambas as fases, portanto esse será o modelo oficial.

### Passo 10, Instanciando e Testando o Pipeline do Modelo:

Resumo do processo de transformação dos passos necessários do notebook para que o algorítimo retorne o ranking de clientes com maiores probabilidades de cancelar de forma “produtizada”.

Parar criar um ambiente de teste do modelo em produção, instanciaremos todas as modificações feitas nos dados em uma mesma função, que irá servir como o script de modificação nos dados crus:

```python
# Função de Transformação e Limpeza

def transform(df):
        
        df = df.sort_values('DATA_PAGAMENTO')

        df['DATA_LIMITE_ENTREGA_PARCEIRO_LOGISTICO'] = pd.to_datetime(df['DATA_LIMITE_ENTREGA_PARCEIRO_LOGISTICO'], format = '%Y-%m-%d %H:%M:%S' , errors = 'coerce')
        df['DATA_PAGAMENTO'] = pd.to_datetime(df['DATA_PAGAMENTO'], format = '%Y-%m-%d %H:%M:%S' , errors = 'coerce')
        df['DATA_APROVACAO_PAGAMENTO'] = pd.to_datetime(df['DATA_APROVACAO_PAGAMENTO'], format = '%Y-%m-%d %H:%M:%S' , errors = 'coerce')
        df['DATA_POSTAGEM'] = pd.to_datetime(df['DATA_POSTAGEM'], format = '%Y-%m-%d %H:%M:%S' , errors = 'coerce')
        df['DATA_ESTIMADA_ENTREGA'] = pd.to_datetime(df['DATA_ESTIMADA_ENTREGA'], format = '%Y-%m-%d %H:%M:%S' , errors = 'coerce')
        df['DATA_ENTREGUE'] = pd.to_datetime(df['DATA_ENTREGUE'], format = '%Y-%m-%d %H:%M:%S' , errors = 'coerce')

        # substitute null's

        df = df[df['DATA_APROVACAO_PAGAMENTO'].notna()]
        df['NOME_CATEGORIA_PRODUTO'].fillna('sem_categoria', inplace=True)
        df['QUANTIA_FOTOS_ANUNCIO'].fillna(0, inplace=True)
        df['COMPRIMENTO_EM_CENTIMETROS'].fillna(26.59, inplace=True)
        df['PESO_EM_GRAMAS'].fillna(1881.23, inplace=True)
        df['LARGURA_PRODUTO_EM_CENTIMETROS'].fillna(20.17, inplace=True)
        df['ALTURA_PRODUTO_EM_CENTIMETROS'].fillna(14.69, inplace=True)

				# fazendo uma cópia após tipagem e limpeza leve
				df_cru = df.copy()

        df['TEMPO_APROVACAO'] = df['DATA_APROVACAO_PAGAMENTO'] - df['DATA_PAGAMENTO']
        df['TEMPO_APROVACAO'] = pd.to_timedelta(df['TEMPO_APROVACAO'])

        df = df.sort_values(by='DATA_PAGAMENTO')
        df['COMPRAS_TOTAIS_ID'] = df.groupby('ID_CLIENTE')['ID_CLIENTE'].cumcount() + 1

        df['POPULARIDADE_VENDEDOR'] = df.groupby('ID_VENDEDOR')['ID_VENDEDOR'].cumcount() + 1

        df['DISTANTE'] = np.where(df['ESTADO_VENDEDOR'] == df['ESTADO_CLIENTE'], 0, 1)

        df['POPULARIDADE_CATEGORIA'] = df.groupby('NOME_CATEGORIA_PRODUTO').cumcount() +1

        df = df.sort_values(['ID_CLIENTE', 'DATA_PAGAMENTO'])
        df['TEMPO_DESDE_ULTIMO_PEDIDO'] = 'sem_pedido_anterior'

        for rep in range(len(df)):
            if df['ID_CLIENTE'].iloc[rep] == df['ID_CLIENTE'].iloc[rep - 1]:
                df['TEMPO_DESDE_ULTIMO_PEDIDO'].iloc[rep] = df['DATA_PAGAMENTO'].iloc[rep] - df['DATA_PAGAMENTO'].iloc[rep-1]

        df['DIA_DA_SEMANA'] = df['DATA_PAGAMENTO'].dt.dayofweek
        df['DIA_DO_MES'] = df['DATA_PAGAMENTO'].dt.day
        df['MES'] = df['DATA_PAGAMENTO'].dt.month
        df['SEMANA_ANO'] = df['DATA_PAGAMENTO'].dt.weekofyear

        df['PREVISAO_DEMORA'] = df['DATA_ESTIMADA_ENTREGA'] - df['DATA_PAGAMENTO']
        df['PREVISAO_DEMORA'] = pd.to_timedelta(df['PREVISAO_DEMORA'])
        
        drop = ['DATA_LIMITE_ENTREGA_PARCEIRO_LOGISTICO', 'DATA_PAGAMENTO', 'DATA_APROVACAO_PAGAMENTO', 'DATA_POSTAGEM',
               'DATA_ESTIMADA_ENTREGA', 'DATA_ENTREGUE', 'PRECO_SEM_FRET']

        # day of week
        df['DIA_DA_SEMANA_sin'] = df['DIA_DA_SEMANA'].apply(lambda x: np.sin(x *(2*np.pi/7)))
        df['DIA_DA_SEMANA_cos'] = df['DIA_DA_SEMANA'].apply(lambda x: np.cos(x *(2*np.pi/7)))

        # month
        df['MES_sin'] = df['MES'].apply(lambda x: np.sin(x *(2*np.pi/12)))
        df['MES_cos'] = df['MES'].apply(lambda x: np.cos(x *(2*np.pi/12)))

        # day
        df['DIA_DO_MES_sin'] = df['DIA_DO_MES'].apply(lambda x: np.sin(x *(2*np.pi/30)))
        df['DIA_DO_MES_cos'] = df['DIA_DO_MES'].apply(lambda x: np.cos(x *(2*np.pi/30)))

        # week of year
        df['SEMANA_ANO_sin'] = df['SEMANA_ANO'].apply(lambda x: np.sin(x *(2*np.pi/52)))
        df['SEMANA_ANO_cos'] = df['SEMANA_ANO'].apply(lambda x: np.cos(x *(2*np.pi/52)))

            
        for i in range(len(df)):
            if df['TEMPO_DESDE_ULTIMO_PEDIDO'].iloc[i] == 'sem_pedido_anterior':
                df['TEMPO_DESDE_ULTIMO_PEDIDO'].iloc[i] = '3650 days 00:00:00'

        df['TEMPO_DESDE_ULTIMO_PEDIDO'] = pd.to_timedelta(df['TEMPO_DESDE_ULTIMO_PEDIDO'])

        # função de conversão de tempo
        def days_hours_minutes(td):
            x = td.days, td.seconds//3600, (td.seconds//60)%60
            x1 = x[0] * 1440
            x2 = x[1] * 60
            x3 = x[2]

            td = x1 + x2 + x3
            return td

        contador = 0

        for rep in range(len(df)):

            df['TEMPO_APROVACAO'].iloc[contador] = days_hours_minutes(df['TEMPO_APROVACAO'].iloc[contador])
            df['PREVISAO_DEMORA'].iloc[contador] = days_hours_minutes(df['PREVISAO_DEMORA'].iloc[contador])
            df['TEMPO_DESDE_ULTIMO_PEDIDO'].iloc[contador] = days_hours_minutes(df['TEMPO_DESDE_ULTIMO_PEDIDO'].iloc[contador])

            contador += 1 
        

        # convertendo inteiro
        df['TEMPO_APROVACAO'] = df['TEMPO_APROVACAO'].astype(int)
        df['PREVISAO_DEMORA'] = df['PREVISAO_DEMORA'].astype(int)
        df['TEMPO_DESDE_ULTIMO_PEDIDO'] = df['TEMPO_DESDE_ULTIMO_PEDIDO'].astype(int)

      
      
      
      df['DIMENSAO'] = df['LARGURA_PRODUTO_EM_CENTIMETROS'] * df['ALTURA_PRODUTO_EM_CENTIMETROS'] * df[
                  'COMPRIMENTO_EM_CENTIMETROS']

        r = RobustScaler()

        df['QUANTIA_ITEMS_NESSA_ORDEM'] = r.fit_transform(df[['QUANTIA_ITEMS_NESSA_ORDEM']].values)
        df['QUANTIA_METODOS_PAGAMENTO'] = r.fit_transform(df[['QUANTIA_METODOS_PAGAMENTO']].values)
        df['QUANTIA_PARCELAS'] = r.fit_transform(df[['QUANTIA_PARCELAS']].values)
        df['PRECO_FRETE'] = r.fit_transform(df[['PRECO_FRETE']].values)
        df['VALOR_COMPRA'] = r.fit_transform(df[['VALOR_COMPRA']].values)
        df['QUANTIA_FOTOS_ANUNCIO'] = r.fit_transform(df[['QUANTIA_FOTOS_ANUNCIO']].values)
        df['PESO_EM_GRAMAS'] = r.fit_transform(df[['PESO_EM_GRAMAS']].values)   
        df['COMPRAS_TOTAIS_ID'] = r.fit_transform(df[['COMPRAS_TOTAIS_ID']].values)  
        df['POPULARIDADE_VENDEDOR'] = r.fit_transform(df[['POPULARIDADE_VENDEDOR']].values)  
        df['POPULARIDADE_CATEGORIA'] = r.fit_transform(df[['POPULARIDADE_CATEGORIA']].values)   
        df['TEMPO_DESDE_ULTIMO_PEDIDO'] = r.fit_transform(df[['TEMPO_DESDE_ULTIMO_PEDIDO']].values)   
        df['TEMPO_APROVACAO'] = r.fit_transform(df[['TEMPO_APROVACAO']].values)  
        df['DIMENSAO'] = r.fit_transform(df[['DIMENSAO']].values)
        df['PREVISAO_DEMORA'] = r.fit_transform(df[['PREVISAO_DEMORA']].values)

        MinMax = MinMaxScaler()

        df['MES'] = MinMax.fit_transform(df[['MES']].values)  
        df['DIA_DA_SEMANA'] = MinMax.fit_transform(df[['DIA_DA_SEMANA']].values)   
        df['DIA_DO_MES'] = MinMax.fit_transform(df[['DIA_DO_MES']].values)
        df['SEMANA_ANO'] = MinMax.fit_transform(df[['SEMANA_ANO']].values)

        cat_cols = df.select_dtypes('O').columns.tolist()
        encoder = OneHotEncoder(top_categories=2, variables=cat_cols, drop_last=False)
        df = encoder.fit_transform(df)

        selected_cols = ['QUANTIA_PARCELAS', 'PRECO_FRETE', 'VALOR_COMPRA',
                         'QUANTIA_FOTOS_ANUNCIO', 'PESO_EM_GRAMAS', 

                 'TEMPO_APROVACAO',

                        'COMPRAS_TOTAIS_ID', 'POPULARIDADE_VENDEDOR', 'POPULARIDADE_CATEGORIA',

                 'TEMPO_DESDE_ULTIMO_PEDIDO',

                        'DIA_DO_MES', 'MES', 'SEMANA_ANO',

                 'PREVISAO_DEMORA', 

                        'DIMENSAO', 'DIA_DO_MES_sin', 'SEMANA_ANO_sin',
                        'SEMANA_ANO_cos', 'METODO_PAGAMENTO_credit_card',
                        'METODO_PAGAMENTO_boleto']
        
        df = df[selected_cols]
        
				# retornar o dataframe para previsão, e o dataframe que entrou
        return df[selected_cols], df_cru
```

Feito, agora traremos um conjunto de dados cru, direto do Kaggle, e pediremos que o modelo preveja esses dados após usar a função de limpeza e correção.

```python
# pegando dataset original
dados = pd.read_csv('olist_valid.csv') 

# retornando dataframe para previsão e dataframe para análise dos resultados
dados, dados_cru = transform(dados) 

# carregando modelo já treinado de arquivo .pickle
rf = pkl.load(open('RandomForest.sav', 'rb'))

# criando ranking com as previsões
previsoes = rf.predict_proba(dados)[:, 1]

# isntanciando previsões à uma coluna no dataset sem modificações de modelagem
dados_cru['Chance_cancelamento'] = previsoes * 100

# arredondando ranking para facilitar
decimals = 2   
dados_cru['Chance_cancelamento'] = dados_cru['Chance_cancelamento'].apply(lambda x: round(x, decimals))

# plotando um mapa com o local dos clientes e suas respectivas chances de cancelar

fig = px.scatter_mapbox(dados_cru, 
                        lat="LAT_CLIENTE", 
                        lon="LNG_CLIENTE",
                        size='VALOR_COMPRA',
                        hover_data=['VALOR_COMPRA', 'CIDADE_CLIENTE'],
                        hover_name='ID_CLIENTE',
                        color='Chance_cancelamento',
                        zoom=3,
                        color_continuous_scale=px.colors.diverging.balance)

fig.update_layout(mapbox_style="carto-positron")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
```

![Untitled 21](https://user-images.githubusercontent.com/85971408/134776356-c40e4cc8-4860-4bb4-8984-6daa9927264c.png)

![Sem_ttulo](https://user-images.githubusercontent.com/85971408/134776365-4eef7eef-934c-4475-b85f-752420532d85.png)


Incrível! Conseguimos fazer um pipeline em que o modelo pega dados como vieram desde o início, transforma, e plota as previsões no mapa com suas respectivas chances de cancelamento!

Espero que tenha gostado, me deixo disponível para qualquer esclarecimento: 41 99999 – 2698.
