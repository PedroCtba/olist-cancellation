import pickle as pkl
import pandas as pd
import numpy as np
import math
import datetime
from feature_engine.encoding import OneHotEncoder
from sklearn.model_selection import train_test_split


class Olist(object):
    def __init__(self):
        
        self.home_path = 'C:/Users/Pedro/Desktop/Codar/Jupyter/Projetos/Olist/'
        
        self.quantia_items_nessa_ordem_scaler = pkl.load(open(self.home_path + 'preparation/transformation_QUANTIA_ITEMS_NESSA_ORDEM.pkl', 'rb'))
        self.quantia_metodos_de_pagamento = pkl.load(open(self.home_path + 'preparation/transformation_QUANTIA_METODOS_PAGAMENTO.pkl', 'rb'))
        self.quantia_parcelas = pkl.load(open(self.home_path + 'preparation/transformation_QUANTIA_PARCELAS.pkl', 'rb'))
        self.preco_frete = pkl.load(open(self.home_path + 'preparation/transformation_PRECO_FRETE.pkl', 'rb'))
        self.valor_compra = pkl.load(open(self.home_path + 'preparation/transformation_VALOR_COMPRA.pkl', 'rb'))
        self.quantia_fotos_anuncio = pkl.load(open(self.home_path + 'preparation/transformation_QUANTIA_FOTOS_ANUNCIO.pkl', 'rb'))
        self.peso_em_gramas = pkl.load(open(self.home_path + 'preparation/transformation_PESO_EM_GRAMAS.pkl', 'rb'))
        self.compras_totais_id = pkl.load(open(self.home_path + 'preparation/transformation_COMPRAS_TOTAIS_ID.pkl', 'rb'))
        self.popularidade_do_vendedor = pkl.load(open(self.home_path + 'preparation/transformation_POPULARIDADE_VENDEDOR.pkl', 'rb'))
        self.popularidade_categoria = pkl.load(open(self.home_path + 'preparation/transformation_POPULARIDADE_CATEGORIA.pkl', 'rb'))
        self.tempo_desde_ultimo_pedido = pkl.load(open(self.home_path + 'preparation/transformation_TEMPO_DESDE_ULTIMO_PEDIDO.pkl', 'rb'))
        self.tempo_aprovacao = pkl.load(open(self.home_path + 'preparation/transformation_TEMPO_APROVACAO.pkl', 'rb'))
        self.dimensao = pkl.load(open(self.home_path + 'preparation/transformation_DIMENSAO.pkl', 'rb'))
        self.previsao_demora = pkl.load(open(self.home_path + 'preparation/transformation_PREVISAO_DEMORA.pkl', 'rb'))

        self.mes = pkl.load(open(self.home_path + 'preparation/transformation_MES.pkl', 'rb'))
        self.dia_da_semana = pkl.load(open(self.home_path + 'preparation/transformation_DIA_DA_SEMANA.pkl', 'rb'))
        self.dia_do_mes = pkl.load(open(self.home_path + 'preparation/transformation_DIA_DO_MES.pkl', 'rb'))
        self.semana_do_ano = pkl.load(open(self.home_path + 'preparation/transformation_SEMANA_ANO.pkl', 'rb'))

        self.encoder = pkl.load(open(self.home_path + 'preparation/transformation_cat_cols.pkl', 'rb'))

    def data_cleaning(self, df):

        # convert types

        df['DATA_LIMITE_ENTREGA_PARCEIRO_LOGISTICO'] = pd.to_datetime(df['DATA_LIMITE_ENTREGA_PARCEIRO_LOGISTICO'],
                                                                      format='%Y-%m-%d %H:%M:%S', errors='coerce')
        df['DATA_PAGAMENTO'] = pd.to_datetime(df['DATA_PAGAMENTO'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        df['DATA_APROVACAO_PAGAMENTO'] = pd.to_datetime(df['DATA_APROVACAO_PAGAMENTO'], format='%Y-%m-%d %H:%M:%S',
                                                        errors='coerce')
        df['DATA_POSTAGEM'] = pd.to_datetime(df['DATA_POSTAGEM'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        df['DATA_ESTIMADA_ENTREGA'] = pd.to_datetime(df['DATA_ESTIMADA_ENTREGA'], format='%Y-%m-%d %H:%M:%S',
                                                     errors='coerce')
        df['DATA_ENTREGUE'] = pd.to_datetime(df['DATA_ENTREGUE'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        df['PREFIXO_CEP_CLIENTE'] = df['PREFIXO_CEP_CLIENTE'].apply(str)
        df['PREFIXO_CEP_VENDEDOR'] = df['PREFIXO_CEP_VENDEDOR'].apply(str)

        # substitute null's

        df = df[df['DATA_APROVACAO_PAGAMENTO'].notna()]
        df['NOME_CATEGORIA_PRODUTO'].fillna('sem_categoria', inplace=True)
        df['QUANTIA_FOTOS_ANUNCIO'].fillna(0, inplace=True)
        df['COMPRIMENTO_EM_CENTIMETROS'].fillna(26.59, inplace=True)
        df['PESO_EM_GRAMAS'].fillna(1881.23, inplace=True)
        df['LARGURA_PRODUTO_EM_CENTIMETROS'].fillna(20.17, inplace=True)
        df['ALTURA_PRODUTO_EM_CENTIMETROS'].fillna(14.69, inplace=True)

        return df

    def feature_enginering(self, df):

        df['TEMPO_APROVACAO'] = df['DATA_APROVACAO_PAGAMENTO'] - df['DATA_PAGAMENTO']
        df['TEMPO_APROVACAO'] = pd.to_timedelta(df['TEMPO_APROVACAO'])

        df = df.sort_values(by='DATA_PAGAMENTO')
        df['COMPRAS_TOTAIS_ID'] = df.groupby('ID_CLIENTE')['ID_CLIENTE'].cumcount() + 1

        df['POPULARIDADE_VENDEDOR'] = df.groupby('ID_VENDEDOR')['ID_VENDEDOR'].cumcount() + 1

        df['DISTANTE'] = np.where(df['ESTADO_VENDEDOR'] == df['ESTADO_CLIENTE'], 0, 1)

        df['POPULARIDADE_CATEGORIA'] = df.groupby('NOME_CATEGORIA_PRODUTO').cumcount() + 1

        df = df.sort_values(['ID_CLIENTE', 'DATA_PAGAMENTO'])
        df['TEMPO_DESDE_ULTIMO_PEDIDO'] = 'sem_pedido_anterior'

        for rep in range(len(df)):
            if df['ID_CLIENTE'].iloc[rep] == df['ID_CLIENTE'].iloc[rep - 1]:
                df['TEMPO_DESDE_ULTIMO_PEDIDO'].iloc[rep] = df['DATA_PAGAMENTO'].iloc[rep] - df['DATA_PAGAMENTO'].iloc[
                    rep - 1]

        df['DIA_DA_SEMANA'] = df['DATA_PAGAMENTO'].dt.dayofweek
        df['DIA_DO_MES'] = df['DATA_PAGAMENTO'].dt.day
        df['MES'] = df['DATA_PAGAMENTO'].dt.month
        df['SEMANA_ANO'] = df['DATA_PAGAMENTO'].dt.weekofyear

        df['PREVISAO_DEMORA'] = df['DATA_ESTIMADA_ENTREGA'] - df['DATA_PAGAMENTO']
        df['PREVISAO_DEMORA'] = pd.to_timedelta(df['PREVISAO_DEMORA'])

        return df

    def data_prep(self, df):

        # instance
        df = df[df['TARGET_STATUS_DA_ORDEM'].isin(['delivered', 'canceled'])]

        drop_cols = [ 'COMPRIMENTO_EM_CENTIMETROS',
                      'LARGURA_PRODUTO_EM_CENTIMETROS', 'ALTURA_PRODUTO_EM_CENTIMETROS', 'DATA_LIMITE_ENTREGA_PARCEIRO_LOGISTICO', 'DATA_PAGAMENTO',
                      'DATA_APROVACAO_PAGAMENTO', 'DATA_POSTAGEM',
                      'DATA_ESTIMADA_ENTREGA', 'DATA_ENTREGUE', 'PRECO_SEM_FRETE']

        # agregando as medidas do produto que tinham muita relação como VOLUME
        df['DIMENSAO'] = df['LARGURA_PRODUTO_EM_CENTIMETROS'] * df['ALTURA_PRODUTO_EM_CENTIMETROS'] * df[
            'COMPRIMENTO_EM_CENTIMETROS']

        df.drop(drop_cols, axis=1, inplace=True)

        df['DIA_DA_SEMANA_sin'] = df['DIA_DA_SEMANA'].apply(lambda x: np.sin(x * (2 * np.pi / 7)))
        df['DIA_DA_SEMANA_cos'] = df['DIA_DA_SEMANA'].apply(lambda x: np.cos(x * (2 * np.pi / 7)))
        df['MES_sin'] = df['MES'].apply(lambda x: np.sin(x * (2 * np.pi / 12)))
        df['MES_cos'] = df['MES'].apply(lambda x: np.cos(x * (2 * np.pi / 12)))
        df['DIA_DO_MES_sin'] = df['DIA_DO_MES'].apply(lambda x: np.sin(x * (2 * np.pi / 30)))
        df['DIA_DO_MES_cos'] = df['DIA_DO_MES'].apply(lambda x: np.cos(x * (2 * np.pi / 30)))
        df['SEMANA_ANO_sin'] = df['SEMANA_ANO'].apply(lambda x: np.sin(x * (2 * np.pi / 52)))
        df['SEMANA_ANO_cos'] = df['SEMANA_ANO'].apply(lambda x: np.cos(x * (2 * np.pi / 52)))

        # definindo pessoas sem pedido anterior como se tivessem pedido há 10 anos
        
        for i in range(len(df)):
             if df['TEMPO_DESDE_ULTIMO_PEDIDO'].iloc[i] == 'sem_pedido_anterior':
                 df['TEMPO_DESDE_ULTIMO_PEDIDO'].iloc[i] = '3650 days 00:00:00'
                
                
        df['TEMPO_DESDE_ULTIMO_PEDIDO'] = pd.to_timedelta(df['TEMPO_DESDE_ULTIMO_PEDIDO'])
        
        c = 1
        if c != 1:
            # função e correção
            def days_hours_minutes(td):
                x = td.days, td.seconds // 3600, (td.seconds // 60) % 60
                x1 = x[0] * 1440
                x2 = x[1] * 60
                x3 = x[2]

                td = x1 + x2 + x3
                return td

            contador = 0

            for rep in range(len(df)):
                df['TEMPO_APROVACAO'].iloc[contador] = days_hours_minutes(df['TEMPO_APROVACAO'].iloc[contador])
                df['PREVISAO_DEMORA'].iloc[contador] = days_hours_minutes(df['PREVISAO_DEMORA'].iloc[contador])
                df['TEMPO_DESDE_ULTIMO_PEDIDO'].iloc[contador] = days_hours_minutes(
                    df['TEMPO_DESDE_ULTIMO_PEDIDO'].iloc[contador])

                contador += 1

            # convertendo inteiro
            df['TEMPO_APROVACAO'] = df['TEMPO_APROVACAO'].astype(int)
            df['PREVISAO_DEMORA'] = df['PREVISAO_DEMORA'].astype(int)
            df['TEMPO_DESDE_ULTIMO_PEDIDO'] = df['TEMPO_DESDE_ULTIMO_PEDIDO'].astype(int)
            
            df['TEMPO_DESDE_ULTIMO_PEDIDO'] = self.tempo_desde_ultimo_pedido.fit_transform(df[['TEMPO_DESDE_ULTIMO_PEDIDO']].values)
            df['TEMPO_APROVACAO'] = self.tempo_aprovacao.fit_transform(df[['TEMPO_APROVACAO']].values)
            df['PREVISAO_DEMORA'] = self.previsao_demora.fit_transform(df[['PREVISAO_DEMORA']].values)
        
        df['QUANTIA_ITEMS_NESSA_ORDEM'] = self.quantia_items_nessa_ordem_scaler.fit_transform(df[['QUANTIA_ITEMS_NESSA_ORDEM']].values)
        df['QUANTIA_METODOS_PAGAMENTO'] = self.quantia_metodos_de_pagamento.fit_transform(df[['QUANTIA_METODOS_PAGAMENTO']].values)
        df['QUANTIA_PARCELAS'] = self.quantia_parcelas.fit_transform(df[['QUANTIA_PARCELAS']].values)
        df['PRECO_FRETE'] = self.preco_frete.fit_transform(df[['PRECO_FRETE']].values)
        df['VALOR_COMPRA'] = self.valor_compra.fit_transform(df[['VALOR_COMPRA']].values)
        df['QUANTIA_FOTOS_ANUNCIO'] = self.quantia_fotos_anuncio.fit_transform(df[['QUANTIA_FOTOS_ANUNCIO']].values)
        df['PESO_EM_GRAMAS'] = self.peso_em_gramas.fit_transform(df[['PESO_EM_GRAMAS']].values)
        df['COMPRAS_TOTAIS_ID'] = self.compras_totais_id.fit_transform(df[['COMPRAS_TOTAIS_ID']].values)
        df['POPULARIDADE_VENDEDOR'] = self.popularidade_do_vendedor.fit_transform(df[['POPULARIDADE_VENDEDOR']].values)
        df['POPULARIDADE_CATEGORIA'] = self.popularidade_categoria.fit_transform(df[['POPULARIDADE_CATEGORIA']].values)
        df['DIMENSAO'] = self.dimensao.fit_transform(df[['DIMENSAO']].values)

         
        df['MES'] = self.mes.fit_transform(df[['MES']].values)
        df['DIA_DA_SEMANA'] = self.dia_da_semana.fit_transform(df[['DIA_DA_SEMANA']].values)
        df['DIA_DO_MES'] = self.dia_do_mes.fit_transform(df[['DIA_DO_MES']].values)
        df['SEMANA_ANO'] = self.semana_do_ano.fit_transform(df[['SEMANA_ANO']].values)
        
        cat_cols = df.select_dtypes('O').columns.tolist()

        df = self.encoder.fit_transform(df)
         
         
        selected_cols = ['QUANTIA_PARCELAS', 'PRECO_FRETE', 'VALOR_COMPRA',
        'QUANTIA_FOTOS_ANUNCIO', 'PESO_EM_GRAMAS', 
        
        # 'TEMPO_APROVACAO',
        
        'COMPRAS_TOTAIS_ID', 'POPULARIDADE_VENDEDOR', 'POPULARIDADE_CATEGORIA',
        
        # 'TEMPO_DESDE_ULTIMO_PEDIDO', 
        
        'DIA_DO_MES', 'MES', 'SEMANA_ANO',
       
        # 'PREVISAO_DEMORA', 
        
        'DIMENSAO', 'DIA_DO_MES_sin', 'SEMANA_ANO_sin',
        'SEMANA_ANO_cos', 'METODO_PAGAMENTO_credit_card',
        'METODO_PAGAMENTO_boleto']
        
        return df[selected_cols]

    def get_prediction(self, model, original_data, test_data):
        # predict
        pred = model.predict(test_data)

        original_data['previsao'] = pred
        
        original_data = original_data.sample(50)
        
        return original_data.to_json(orient='records', date_format='iso')