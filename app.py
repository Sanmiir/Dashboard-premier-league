import streamlit as st
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(
    page_title="Dashboard Premier League",
    page_icon="⚽",
    layout="wide" 
)

@st.cache_data
def carregar_dados(caminho):
    print("Executando a carga e limpeza dos dados (só vai acontecer na primeira vez)...")
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
    df_final['Shot_Efficiency'].fillna(media_eficiencia, inplace=True)
    
    def definir_resultado(row):
        if row['Goals'] > row['Opponent_Goals']:
            return 'Vitória'
        elif row['Goals'] < row['Opponent_Goals']:
            return 'Derrota'
        else:
            return 'Empate'
    df_final['Tipo_Resultado'] = df_final.apply(definir_resultado, axis=1)

    print("Carga e limpeza concluídas!")
    return df_final

df = carregar_dados('archive/*.csv')

if df.empty:
    st.error("Nenhum arquivo CSV encontrado na pasta 'archive'. Verifique a estrutura de pastas.")
    st.stop()

st.title('⚽ Análise de Dados do Big 6 da Premier League (2013-2025)')
st.markdown("Utilize os filtros na barra lateral para uma análise detalhada do desempenho dos times.")

st.sidebar.header('Filtros Interativos')

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

tab1, tab2, tab3, tab4 = st.tabs(["📊 Visão Geral", "📈 Desempenho Detalhado", "🔍 Análise de Correlação", "💾 Dados Brutos"])

with tab1:
    st.header("Resumo Geral dos Times Selecionados")
    st.markdown(f"Analisando dados de **{df_filtrado['Season'].nunique()}** temporadas (de {temporadas_selecionadas[0]} a {temporadas_selecionadas[1]}).")
    
    total_jogos = df_filtrado.shape[0]
    total_gols = int(df_filtrado['Goals'].sum())
    media_gols_jogo = df_filtrado['Goals'].mean()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Jogos Analisados", f"{total_jogos:,}".replace(",", "."))
    col2.metric("Total de Gols Marcados", f"{total_gols:,}".replace(",", "."))
    col3.metric("Média de Gols por Jogo", f"{media_gols_jogo:.2f}")

    st.subheader('Distribuição de Resultados')
    contagem_resultados = df_filtrado['Tipo_Resultado'].value_counts()
    fig_pie = px.pie(
        values=contagem_resultados.values, 
        names=contagem_resultados.index, 
        title='Proporção de Vitórias, Derrotas e Empates',
        color=contagem_resultados.index,
        color_discrete_map={'Vitória':'#4CAF50', 'Derrota':'#F44336', 'Empate':'#FFC107'}
    )
    st.plotly_chart(fig_pie, use_container_width=True)


with tab2:
    st.header("Análise de Desempenho Ofensivo e Defensivo")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Balanço Ataque vs. Defesa')
        df_balanco = df_filtrado.groupby('time')[['Goals', 'Opponent_Goals']].mean().sort_values(by='Goals', ascending=False)
        df_balanco = df_balanco.rename(columns={'Goals': 'Gols Marcados', 'Opponent_Goals': 'Gols Sofridos'})
        st.bar_chart(df_balanco)
        st.markdown("Comparativo da média de gols marcados versus a média de gols sofridos por jogo.")

    with col2:
        st.subheader('Eficiência de Finalização')
        df_filtrado_com_chutes = df_filtrado[df_filtrado['Shots_On_Target'] > 0]
        df_eficiencia = df_filtrado_com_chutes.groupby('time').apply(
            lambda x: x['Goals'].sum() / x['Shots_On_Target'].sum()
        ).sort_values(ascending=False)
        df_eficiencia = df_eficiencia.rename("Gols por Chute no Alvo")
        st.bar_chart(df_eficiencia)
        st.markdown("Mede quantos gols um time marca para cada chute no alvo. Um indicador de precisão.")

    st.subheader('Evolução da Média de Gols por Temporada')
    df_evolucao = df_filtrado.groupby(['Season', 'time'])['Goals'].mean().reset_index()
    fig_line = px.line(
        df_evolucao,
        x='Season',
        y='Goals',
        color='time',
        title='Média de Gols Marcados por Temporada',
        markers=True
    )
    st.plotly_chart(fig_line, use_container_width=True)

with tab3:
    st.header('Análise de Correlação')
    st.markdown("O mapa de calor abaixo mostra como as diferentes métricas se relacionam.")
    colunas_interesse = ['Goals', 'Possession', 'Shots', 'Shots_On_Target', 'Pass_Accuracy', 'Fouls']
    colunas_existentes = [col for col in colunas_interesse if col in df_filtrado.columns]
    df_correlacao = df_filtrado[colunas_existentes].corr()
    
    fig_heatmap, ax_heatmap = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_correlacao, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_heatmap)
    st.pyplot(fig_heatmap)

with tab4:
    st.header('Dados Brutos e Download')
    st.markdown("Visualize os dados filtrados abaixo ou faça o download do conjunto completo.")
    st.dataframe(df_filtrado)
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
       label="Baixar todos os dados como CSV",
       data=csv,
       file_name='dados_completos_premier_league.csv',
       mime='text/csv',
    )