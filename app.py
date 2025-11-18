import streamlit as st
import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, confusion_matrix, silhouette_score

st.set_page_config(
    page_title="Dashboard Premier League & ML",
    page_icon="⚽",
    layout="wide" 
)

@st.cache_data
def carregar_dados(caminho):
    print("Executando a carga e limpeza dos dados...")
    arquivos_csv = glob.glob(caminho)
    lista_de_dataframes = []

    for nome_arquivo in arquivos_csv:
        try:
            df_temporario = pd.read_csv(nome_arquivo, encoding='utf-8')
        except UnicodeDecodeError:
            df_temporario = pd.read_csv(nome_arquivo, encoding='latin1')

        nome_time = os.path.basename(nome_arquivo).replace('.csv', '')
        df_temporario['time'] = nome_time
        lista_de_dataframes.append(df_temporario)

    if not lista_de_dataframes:
        return pd.DataFrame() 

    df_final = pd.concat(lista_de_dataframes, ignore_index=True)

    df_final['Date'] = pd.to_datetime(df_final['Date'])
    df_final['Season'] = df_final['Date'].dt.year
    
    media_eficiencia = df_final['Shot_Efficiency'].mean()
    df_final['Shot_Efficiency'] = df_final['Shot_Efficiency'].fillna(media_eficiencia)
    
    def definir_resultado(row):
        if row['Goals'] > row['Opponent_Goals']:
            return 'Vitória'
        elif row['Goals'] < row['Opponent_Goals']:
            return 'Derrota'
        else:
            return 'Empate'
    df_final['Tipo_Resultado'] = df_final.apply(definir_resultado, axis=1)

    return df_final

@st.cache_resource
def treinar_modelos(df):
    resultados = {}
    
   
    X_reg = df[['Shots_On_Target', 'Possession']]
    y_reg = df['Goals']
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    model_reg = LinearRegression()
    model_reg.fit(X_train_reg, y_train_reg)
    y_pred_reg = model_reg.predict(X_test_reg)
    
    resultados['regressao'] = {
        'mae': mean_absolute_error(y_test_reg, y_pred_reg),
        'r2': r2_score(y_test_reg, y_pred_reg),
        'y_test': y_test_reg,
        'y_pred': y_pred_reg
    }

    
    le = LabelEncoder()
    df['Resultado_Cod'] = le.fit_transform(df['Tipo_Resultado'])
    classes = le.classes_
    
    X_cls = df[['Possession', 'Shots_On_Target', 'Pass_Accuracy', 'Fouls']]
    y_cls = df['Resultado_Cod']
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
    
    model_cls = RandomForestClassifier(n_estimators=100, random_state=42)
    model_cls.fit(X_train_c, y_train_c)
    y_pred_cls = model_cls.predict(X_test_c)
    
    resultados['classificacao'] = {
        'acuracia': accuracy_score(y_test_c, y_pred_cls),
        'matriz': confusion_matrix(y_test_c, y_pred_cls),
        'classes': classes
    }

   
    X_cluster = df[['Possession', 'Shots', 'Fouls']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    

    df_cluster = df.copy()
    df_cluster['Cluster'] = clusters
    
    resultados['agrupamento'] = {
        'silhouette': silhouette_score(X_scaled, clusters),
        'df_cluster': df_cluster,
        'perfil': df_cluster.groupby('Cluster')[['Possession', 'Shots', 'Fouls']].mean()
    }
    
    return resultados

df = carregar_dados('archive/*.csv')

if df.empty:
    st.error("Nenhum arquivo CSV encontrado na pasta 'archive'.")
    st.stop()


resultados_ml = treinar_modelos(df)


st.title('⚽ Análise Avançada da Premier League (Data Science & ML)')
st.markdown("Dashboard completo com Estatística Descritiva e Machine Learning.")

st.sidebar.header('Filtros (Visualização)')

lista_times = sorted(df['time'].unique())
times_selecionados = st.sidebar.multiselect(
    'Selecione os times:',
    options=lista_times,
    default=lista_times[:3]
)

min_season = int(df['Season'].min())
max_season = int(df['Season'].max())
temporadas_selecionadas = st.sidebar.slider(
    'Selecione o intervalo de temporadas:',
    min_value=min_season,
    max_value=max_season,
    value=(min_season, max_season) 
)


if times_selecionados:
    df_filtrado = df[
        (df['time'].isin(times_selecionados)) &
        (df['Season'].between(temporadas_selecionadas[0], temporadas_selecionadas[1]))
    ]
else:
    df_filtrado = df[df['Season'].between(temporadas_selecionadas[0], temporadas_selecionadas[1])]


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Visão Geral", 
    "📈 Desempenho Detalhado", 
    "🔍 Correlação", 
    "🤖 Machine Learning",
    "💾 Dados Brutos"
])


with tab1:
    st.header("Resumo Geral")
    
    total_jogos = df_filtrado.shape[0]
    total_gols = int(df_filtrado['Goals'].sum())
    media_gols_jogo = df_filtrado['Goals'].mean()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Jogos Analisados", f"{total_jogos:,}")
    col2.metric("Gols Totais", f"{total_gols:,}")
    col3.metric("Média de Gols/Jogo", f"{media_gols_jogo:.2f}")

    st.subheader('Distribuição de Resultados')
    contagem = df_filtrado['Tipo_Resultado'].value_counts()
    fig_pie = px.pie(values=contagem.values, names=contagem.index, color=contagem.index,
                     color_discrete_map={'Vitória':'#4CAF50', 'Derrota':'#F44336', 'Empate':'#FFC107'})
    st.plotly_chart(fig_pie, use_container_width=True)


with tab2:
    st.header("Ofensivo vs Defensivo")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Balanço de Gols')
        df_balanco = df_filtrado.groupby('time')[['Goals', 'Opponent_Goals']].mean().sort_values(by='Goals', ascending=False)
        st.bar_chart(df_balanco)
    
    with col2:
        st.subheader('Eficiência de Chute')
        df_eff = df_filtrado[df_filtrado['Shots_On_Target'] > 0]
        df_eff = df_eff.groupby('time').apply(lambda x: x['Goals'].sum() / x['Shots_On_Target'].sum()).sort_values(ascending=False)
        st.bar_chart(df_eff)

    st.subheader('Evolução Gols/Temporada')
    df_evo = df_filtrado.groupby(['Season', 'time'])['Goals'].mean().reset_index()
    fig_line = px.line(df_evo, x='Season', y='Goals', color='time', markers=True)
    st.plotly_chart(fig_line, use_container_width=True)


with tab3:
    st.header('Correlação de Variáveis')
    cols = ['Goals', 'Possession', 'Shots', 'Shots_On_Target', 'Pass_Accuracy', 'Fouls']
    cols_exists = [c for c in cols if c in df_filtrado.columns]
    
    fig_heat, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_filtrado[cols_exists].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig_heat)


with tab4:
    st.header("Resultados dos Modelos de IA")
    st.info("Nota: Os modelos foram treinados com o dataset COMPLETO para garantir maior precisão estatística.")

   
    st.subheader("1. Regressão Linear: Previsão de Gols")
    res_reg = resultados_ml['regressao']
    
    col1, col2 = st.columns(2)
    col1.metric("R² (Explicação)", f"{res_reg['r2']:.2f}")
    col2.metric("Erro Médio Absoluto (MAE)", f"{res_reg['mae']:.2f}")
    
    fig_reg, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(res_reg['y_test'], res_reg['y_pred'], alpha=0.5, color='blue')
    ax.plot([0, 10], [0, 10], 'k--', lw=2)
    ax.set_xlabel('Gols Reais')
    ax.set_ylabel('Gols Previstos')
    ax.set_title('Real vs Previsto')
    st.pyplot(fig_reg)

    st.divider()

  
    st.subheader("2. Classificação: Resultado do Jogo")
    res_cls = resultados_ml['classificacao']
    
    st.metric("Acurácia do Modelo", f"{res_cls['acuracia']:.2%}")
    
    fig_conf, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(res_cls['matriz'], annot=True, fmt='d', cmap='Blues', 
                xticklabels=res_cls['classes'], yticklabels=res_cls['classes'], ax=ax)
    ax.set_title('Matriz de Confusão')
    st.pyplot(fig_conf)

    st.divider()

    st.subheader("3. Agrupamento (K-Means): Estilos de Jogo")
    res_cl = resultados_ml['agrupamento']
    
    st.metric("Silhouette Score", f"{res_cl['silhouette']:.2f}")
  
    fig_cluster = px.scatter(
        res_cl['df_cluster'], 
        x='Possession', 
        y='Shots', 
        color=res_cl['df_cluster']['Cluster'].astype(str),
        title='Clusters de Estilo de Jogo (Posse vs Chutes)',
        color_discrete_sequence=px.colors.qualitative.Set1,
        hover_data=['time', 'Season', 'Result']
    )
    st.plotly_chart(fig_cluster, use_container_width=True)
    
    st.markdown("**Perfil Médio dos Grupos:**")
    st.dataframe(res_cl['perfil'])

with tab5:
    st.dataframe(df_filtrado)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Baixar CSV", csv, "dados_premier_league.csv", "text/csv")
