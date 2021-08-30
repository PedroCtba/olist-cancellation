CREATE TABLE OLIST_GERAL AS SELECT * FROM
(
SELECT
	ORDENS.order_id AS ID_ORDEM,
	CLIENTES.customer_unique_id AS ID_CLIENTE,
	ORDEM_ITEM.product_id AS ID_PRODUTO,
	ORDEM_ITEM.seller_id AS ID_VENDEDOR,
	ORDEM_ITEM.shipping_limit_date AS DATA_LIMITE_ENTREGA_PARCEIRO_LOGISTICO,
	ORDENS.order_purchase_timestamp AS DATA_PAGAMENTO,
	ORDENS.order_approved_at AS DATA_APROVACAO_PAGAMENTO,
	ORDENS.order_delivered_carrier_date AS DATA_POSTAGEM,
	ORDENS.order_estimated_delivery_date AS DATA_ESTIMADA_ENTREGA,
	ORDENS.order_delivered_customer_date AS DATA_ENTREGUE,
	ORDEM_ITEM.order_item_id AS QUANTIA_ITEMS_NESSA_ORDEM,
	PAGAMENTOS.payment_sequential AS QUANTIA_METODOS_PAGAMENTO,
	PAGAMENTOS.payment_type AS METODO_PAGAMENTO,
	PAGAMENTOS.payment_installments AS QUANTIA_PARCELAS,
	ORDEM_ITEM.price AS PRECO_SEM_FRETE,
	ORDEM_ITEM.freight_value AS PRECO_FRETE,
	PAGAMENTOS.payment_value AS VALOR_COMPRA,
	PRODUTOS.product_category_name AS NOME_CATEGORIA_PRODUTO,
	PRODUTOS.product_photos_qty AS QUANTIA_FOTOS_ANUNCIO,
	PRODUTOS.product_weight_g AS PESO_EM_GRAMAS,
	PRODUTOS.product_length_cm AS COMPRIMENTO_EM_CENTIMETROS,
	PRODUTOS.product_width_cm AS LARGURA_PRODUTO_EM_CENTIMETROS,
	PRODUTOS.product_height_cm AS ALTURA_PRODUTO_EM_CENTIMETROS,
	CLIENTES.customer_zip_code_prefix AS PREFIXO_CEP_CLIENTE,
	CLIENTES.customer_city AS CIDADE_CLIENTE,
	CLIENTES.customer_state AS ESTADO_CLIENTE,
	VENDEDORES.seller_zip_code_prefix AS PREFIXO_CEP_VENDEDOR,
	VENDEDORES.seller_city AS CIDADE_VENDEDOR,
	VENDEDORES.seller_state AS ESTADO_VENDEDOR,
	ORDENS.order_status AS TARGET_STATUS_DA_ORDEM
FROM olist_orders_dataset AS ORDENS
	INNER JOIN olist_order_payments_dataset PAGAMENTOS ON ORDENS.order_id = PAGAMENTOS.order_id
	INNER JOIN olist_customers_dataset CLIENTES ON ORDENS.customer_id = CLIENTES.customer_id
	INNER JOIN olist_order_items_dataset ORDEM_ITEM ON ORDENS.order_id = ORDEM_ITEM.order_id
	INNER JOIN olist_products_dataset PRODUTOS ON ORDEM_ITEM.product_id = PRODUTOS.product_id
	INNER JOIN olist_sellers_dataset VENDEDORES ON ORDEM_ITEM.seller_id = VENDEDORES.seller_id
) AS OLIST_GERAL