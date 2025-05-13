#!/usr/bin/env python
# coding: utf-8

# In[1]:

from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import bokeh
from math import pi
from bokeh.plotting import figure, output_file, save
from bokeh.models import (ColumnDataSource, Plot, AnnularWedge, Legend, 
                         LegendItem, Range1d, HoverTool)
from bokeh.layouts import column
from bokeh.io import export_png
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
import time

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import os
import json
from jinja2 import Template
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import numpy as np


# AJUSTE E FORMATAÇÃO DOS DADOS

# In[3]:

st.set_page_config(layout="wide")

#DICIONARIOS
##dicionário classificação
classificacao_rochas = {
    "Urubici": {
        "SiO2": (49, float('inf')),
        "TiO2": (3.3, float('inf')),
        "P2O5": (0.45, float('inf')),
        "Fe2O3t": (0, 14.5),
        "Sr": (550, float('inf')),
        "Ba": (500, float('inf')),
        "Zr": (250, float('inf')),
        "Ti/Zr": (57, float('inf')),
        "Ti/Itrio": (500, float('inf')),
        "Zr/Itrio": (6.5, float('inf')),
        "Sr/Itrio": (14, float('inf')),
        "Ba/Itrio": (14, float('inf')),
    },
    "Pitanga": {
        "SiO2": (47, float('inf')),
        "TiO2": (2.8, float('inf')),
        "P2O5": (0.35, float('inf')),
        "Fe2O3t": (12.5, 18),
        "Sr": (350, float('inf')),
        "Ba": (200, float('inf')),
        "Zr": (200, float('inf')),
        "Ti/Zr": (60, float('inf')),
        "Ti/Itrio": (350, float('inf')),
        "Zr/Itrio": (5.5, float('inf')),
        "Sr/Itrio": (8, float('inf')),
        "Ba/Itrio": (9, float('inf')),
    },
    "Paranapanema": {
        "SiO2": (48, 53),
        "TiO2": (1.7, 3.2),
        "P2O5": (0.2, 0.8),
        "Fe2O3t": (12.5, 17),
        "Sr": (200, 450),
        "Ba": (200, 650),
        "Zr": (120, 250),
        "Ti/Zr": (65, float('inf')),
        "Ti/Itrio": (350, float('inf')),
        "Zr/Itrio": (4.0, 7.0),
        "Sr/Itrio": (4.5, 15),
        "Ba/Itrio": (5.0, 19),
    },
    "Ribeira": {
        "SiO2": (49, 52),
        "TiO2": (1.5, 2.3),
        "P2O5": (0.15, 0.50),
        "Fe2O3t": (12.0, 16),
        "Sr": (200, 375),
        "Ba": (200, 600),
        "Zr": (100, 200),
        "Ti/Zr": (65, float('inf')),
        "Ti/Itrio": (300, float('inf')),
        "Zr/Itrio": (3.5, 7.0),
        "Sr/Itrio": (5.0, 17),
        "Ba/Itrio": (6.0, 19),
    },
    "Esmeralda": {
        "SiO2": (48, 55),
        "TiO2": (1.1, 2.3),
        "P2O5": (0.1, 0.35),
        "Fe2O3t": (12.0, 17),
        "Sr": (0, 250),
        "Ba": (90, 400),
        "Zr": (65, 210),
        "Ti/Zr": (60, float('inf')),
        "Ti/Itrio": (0, 330),
        "Zr/Itrio": (2.0, 5.0),
        "Sr/Itrio": (0, 9),
        "Ba/Itrio": (0, 12),
    },
    "Gramado": {
        "SiO2": (49, 60),
        "TiO2": (0.7, 2.0),
        "P2O5": (0.05, 0.40),
        "Fe2O3t": (9.0, 16),
        "Sr": (140, 400),
        "Ba": (100, 700),
        "Zr": (65, 275),
        "Ti/Zr": (0, 70),
        "Ti/Itrio": (0, 330),
        "Zr/Itrio": (3.5, 6.5),
        "Sr/Itrio": (0, 13),
        "Ba/Itrio": (0, 19),
    }
}
#dicinário cores
colors = {
    'esmeralda': '#FF0000',        # Vermelho
    'gramado': '#FFA500',          # Laranja
    'ribeira': '#FFFF00',          # Amarelo
    'paranapanema': '#0000FF',     # Azul
    'pitanga': '#008000',          # Verde
    'urubici': '#800080',          # Roxo
    'não classificado': '#D3D3D3'  # Cinza claro
}


# In[4]:


#importar e transformar para float
arquivo = Path(".").resolve() / "Amostras_completo.csv"
amostras_df = pd.read_csv(arquivo, sep=";")
amostras_df = amostras_df.rename(columns={'Fe2O3(t)': 'Fe2O3t'})
colunas_numericas = ['SiO2', 'MgO', 'TiO2', 'Al2O3', 'Fe2O3t', 'MnO','CaO', 'Na2O', 'K2O', 'P2O5', 'Fe2O3', 'FeO', 'Ba', 'Rb','Sr', 'Itrio', 'Zr', 'Ti']
for coluna in colunas_numericas:
    amostras_df[coluna] = pd.to_numeric(amostras_df[coluna], errors='coerce')
print(amostras_df.dtypes)


# In[5]:


#Configuração de caminhos
base_path = r'C:\Users\LabMEG_09\Downloads\PMP_produtos\\'
os.makedirs(base_path, exist_ok=True)


# In[6]:


#FUNÇÕES
##classificação
def classificar_amostra(amostra):
    for tipo_rocha, limites in classificacao_rochas.items():
        if all(limites[elemento][0] <= amostra[elemento] <= limites[elemento][1] for elemento in limites):
            return tipo_rocha
    return "não classificado"
def verificar_criterios_classificacao(row):
    motivos = []
    
    for coluna in colunas_necessarias:
        valor = row[coluna]
        
        # Verifica se o valor é NaN ou uma string vazia ou contém espaços em branco
        if pd.isnull(valor) or (isinstance(valor, str) and valor.strip() == ''):
            motivos.append(f"{coluna} está ausente")
    
    return ', '.join(motivos) if motivos else "Classificada"
#cores
def get_color(classificacao):
    # Converte para minúsculas e remove espaços extras
    key = classificacao.lower().strip()
    return colors.get(key, colors['não classificado'])
#exportação PNG 
def export_to_png(plot, filename):
    options = Options()
    options.headless = True
    driver = webdriver.Firefox(options=options)
    try:
        html_path = base_path + filename.replace('.png', '.html')
        output_file(html_path)
        save(plot)
        driver.get("file:///" + html_path)
        time.sleep(2)
        export_png(plot, filename=base_path + filename, webdriver=driver)
    finally:
        driver.quit()


# CLASSFICAÇÃO INICIAL

# In[8]:


#razões
amostras_df['Sr/Itrio'] = amostras_df['Sr'] / amostras_df['Itrio']
amostras_df['Ba/Itrio'] = amostras_df['Ba'] / amostras_df['Itrio']
amostras_df['Ti/Itrio'] = amostras_df['Ti'] / amostras_df['Itrio']
amostras_df['Ti/Zr'] = amostras_df['Ti'] / amostras_df['Zr']
amostras_df['Zr/Itrio'] = amostras_df['Zr'] / amostras_df['Itrio']
amostras_df


# In[9]:


#df classificação peate
colunas_necessarias = ['SiO2', 'TiO2', 'P2O5', 'Fe2O3t', 'Sr', 'Ba', 'Zr', 'Ti/Itrio', 'Ti/Zr', 'Zr/Itrio', 'Sr/Itrio', 'Ba/Itrio']
amostras_df = amostras_df.dropna(subset=colunas_necessarias)
amostras_df


# In[10]:


#criar coluna classificação
amostras_df['Classificacao'] = amostras_df.apply(classificar_amostra, axis=1)
#criar coluna de contagem 
contagem = amostras_df['Classificacao'].value_counts()
porcentagem = (contagem / contagem.sum()) * 100
resultado = pd.DataFrame({
    'Classificacao': contagem.index,
    'Contagem': contagem.values,
    'Porcentagem': porcentagem.round(2).values,
    'Cores': [get_color(x) for x in contagem.index]  
})
print(resultado)
#exportar excel
amostras_df.to_excel(r'C:\Users\LabMEG_09\Downloads\amostras_classificadas2.xlsx', index=False) #exportar


# In[11]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import plotly.graph_objects as go


# DASHBOARD

# In[13]:


import qrcode
import os
from datetime import datetime

def generate_qr_code(url, output_path):
    """Gera um QR code para a URL especificada"""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)

    qr_img = qr.make_image(fill_color="black", back_color="white")
    qr_img.save(output_path)
    return output_path

def main():
    # Diretório base
    base_path = r'C:\Users\LabMEG_09\Downloads\PMP_produtos\\'
    
    # Nome do arquivo do dashboard
    dashboard_filename = "dashboard_final.html"
    
    # Timestamp para nome único do QR code
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    qr_filename = f"qr_code_{timestamp}.png"
    qr_path = os.path.join(base_path, qr_filename)
    
    # URL exemplo - você deve substituir pela URL real após hospedar
    url_exemplo = "https://github.com/MaurenGaspar/PMP#/dashboard_final.html"
    
    # Gerar QR code
    qr_path = generate_qr_code(url_exemplo, qr_path)
    
    print("\nInstruções para disponibilizar o dashboard publicamente:")
    print("1. Crie um repositório no GitHub (por exemplo: pmp-dashboard)")
    print("2. Faça upload do arquivo dashboard_final.html para o repositório")
    print("3. Nas configurações do repositório, ative o GitHub Pages:")
    print("   - Vá em Settings > Pages")
    print("   - Selecione a branch main e pasta root")
    print("   - Clique em Save")
    print("4. Após alguns minutos, seu dashboard estará disponível em:")
    print("   https://seu-usuario.github.io/pmp-dashboard/dashboard_final.html")
    print(f"\nQR Code gerado em: {qr_path}")
    print("5. Atualize este script com a URL real e gere o QR code novamente")

if __name__ == "__main__":
    main() 


# In[14]:


def create_donut_chart(df):
    # Contar ocorrências de cada classificação
    counts = df['Classificacao'].value_counts()
    
    # Criar o gráfico de rosca
    fig = go.Figure(data=[go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.4,  # Tamanho do buraco central (0.4 = 40%)
        marker_colors=[colors.get(cls, colors['não classificado']) for cls in counts.index],
        textinfo='label+percent',
        insidetextorientation='radial'
    )])
    
    # Ajustar layout
    fig.update_layout(
        title_text="Distribuição por Classificação",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# E na chamada da função:
donut_fig = create_donut_chart(amostras_df)  # Note o nome corrigido da função

#criar o Gráfico de Barras
def create_bar_chart():
    source = ColumnDataSource(resultado)
    
    p = figure(
        x_range=resultado['Classificacao'].tolist(),
        title="Count by Classification",
        x_axis_label="Classificação",
        y_axis_label="Porcentagem (%)",
        height=400,
        width=800,
        tools="hover,pan,wheel_zoom,reset"
    )
    
    p.vbar(
        x='Classificacao',
        top='Porcentagem',
        width=0.7,
        source=source,
        fill_color='Cores',
        line_color="white"
    )
    
    hover = p.select_one(HoverTool)
    hover.tooltips = [
        ("Classificação", "@Classificacao"),
        ("Contagem", "@Contagem"),
        ("Porcentagem", "@Porcentagem%")
    ]
    return p

#criar mapa
def create_geo_map():
    # Criar o mapa com as cores padrão do projeto
    fig = px.scatter_map(amostras_df,
                            lat="LATf",
                            lon="LONGf",
                            color="Classificacao",
                            color_discrete_map=colors,  # Usando seu dicionário de cores
                            zoom=4,
                            height=600,
                            title="Distribuição Geográfica por Suíte",
                            hover_name="Magma types")

    # Configurações do layout do mapa
    fig.update_layout(
        mapbox_style="carto-positron",
        legend_title_text='Magma types',  # Adicione esta linha
        mapbox_layers=[
            {
                "below": 'traces',
                "sourcetype": "raster",
                "source": [
                    "https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}"
                ],
                "opacity": 0.3
            }
        ],
        margin={"r":0,"t":40,"l":0,"b":0}
    )
    
    # Exportar o mapa
    caminho_exportacao = base_path + "mapa_suites.html"
    fig.write_html(caminho_exportacao)
    
    print(f"Mapa exportado com sucesso para: {caminho_exportacao}")
    return fig



#Criar o Dashboard
def create_dashboard():
    donut = create_donut()
    bars = create_bar_chart()
    mapdash= create_geo_map()
    
    dashboard = column(donut, bars, mapdash)
    
    # Exportar HTML do dashboard
    output_file(base_path + "dashboard.html")
    save(dashboard)
    
    # Exportar PNGs individuais (opcional)
    export_to_png(donut, "donut.png")
    export_to_png(bars, "barras.png")
    export_to_png(mapdash, "mapa.png")
    
    print("Dashboard criado com sucesso!")
    print(f"- Visualização completa: {base_path}dashboard.html")
    print(f"- Donut (PNG): {base_path}donut.png")
    print(f"- Barras (PNG): {base_path}barras.png")


# In[30]:


DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Dashboard PMP</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
            padding: 20px;
            min-height: 100vh;
        }

        .header {
            text-align: center;
            padding: 20px;
            background-color: white;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }

        .header h1 {
            color: #333;
            font-size: 24px;
            margin-bottom: 10px;
        }

        .header h2 {
            color: #666;
            font-size: 16px;
            font-style: italic;
        }
        
        .dashboard-container {
            max-width: 1600px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            padding: 10px;
            background-color: transparent;
            border-radius: 10px;
            box-shadow: none;
        }
        
        .map-container {
            grid-column: 1;
            grid-row: 1 / span 2;
            min-height: 500px;
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            border: none;
        }
        
        .chart-container {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 1px;
            background: white;
            min-height: 250px;
            border: none;
        }
        
        .pca-container {
            grid-column: 1 / span 2;
            grid-row: 4;
            min-height: 400px;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 5px;
            background: white;
            border: none;
        }
        
        .plotly-graph {
            width: 100%;
            height: 100%;
        }
        
        @media (max-width: 1200px) {
            .dashboard-container {
                grid-template-columns: 1fr;
            }
            
            .map-container,
            .pca-container {
                grid-column: 1;
            }
            
            .chart-container {
                min-height: 300px;
            }
        }
    </style>
</head>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Lithogeochemical Classification</title>
    <style>
        body {
            font-family: "Times New Roman", Times, serif;
            margin: 40px;
            background-color: #ffffff;
            color: #333;
        }

        h1 {
            font-size: 32px;
            text-align: center;
            margin-bottom: 10px;
        }

        h2 {
            font-size: 20px;
            text-align: center;
            margin-top: 0;
            margin-bottom: 30px;
            font-style: italic;
        }

        .intro {
            max-width: 800px;
            margin: auto;
            font-size: 18px;
            line-height: 1.6;
            text-align: justify;
        }

        .reference {
            font-size: 16px;
            margin-top: 20px;
        }
    </style>
</head>
<body>

    <h1>Lithogeochemical Classification of the Paraná Magmatic Province</h1>
    <h2>Mauren Gaspar (UFRJ/UFRRJ) & Thais Maia (USP)</h2>

    <div class="intro">
        <p>
            Interactive visualization of lithogeochemical data classified using the Peate (1992) methodology. 
            This tool features a geographical map of sample distribution and dynamic statistical plots, 
            enabling integrated interpretation of geochemical signatures.
        </p>
        <p class="reference">
            Based on: Peate, D.W., Hawkesworth, C. J., & Mantovani, M. S. M. 
            <em>Chemical stratigraphy of the Paraná lavas (South America): classification of magma types and their spatial distribution.</em> 
            Bulletin of Volcanology, 55(1–2), 119–139, 1992.  
            Available at: <a href="https://doi.org/10.1007/bf00301125" target="_blank">https://doi.org/10.1007/bf00301125</a>
        </p>
    </div>
    
    <div class="dashboard-container">
        <div class="map-container">
            <div id="map" class="plotly-graph"></div>
        </div>
        <div class="chart-container">
            <div id="donut" class="plotly-graph"></div>
        </div>
        <div class="chart-container">
            <div id="bar" class="plotly-graph"></div>
        </div>
        <div class="chart-container">
            <div id="correlation" class="plotly-graph"></div>
        </div>
        <div class="pca-container">
            <div id="pca" class="plotly-graph"></div>
        </div>
    </div>
    
    <script>
        const INITIAL_DATA = {
            allData: {{ all_data | safe }},
            colorMap: {{ colors | tojson }},
            numericColumns: {{ numeric_columns | tojson }},
            mapData: {{ map_json | safe }},
            donutData: {{ donut_json | safe }},
            barData: {{ bar_json | safe }},
            correlationData: {{ correlation_json | safe }},
            pcaData: {{ pca_json | safe }}
        };

        let currentBounds = null;

        function getCurrentBounds() {
            const mapDiv = document.getElementById('map');
            if (!mapDiv.layout || !mapDiv.layout.mapbox) return null;
            const zoom = mapDiv.layout.mapbox.zoom;
            const center = mapDiv.layout.mapbox.center;
            const latDiff = 180 / Math.pow(2, zoom);
            const lonDiff = 360 / Math.pow(2, zoom);
            return {
                north: center.lat + latDiff,
                south: center.lat - latDiff,
                east: center.lon + lonDiff,
                west: center.lon - lonDiff
            };
        }

        function getVisibleData() {
            const mapDiv = document.getElementById('map');
            if (!mapDiv.data || !mapDiv.data[0]) return INITIAL_DATA.allData;
            if (mapDiv.data[0].visible === 'legendonly') return [];
            currentBounds = currentBounds || getCurrentBounds();
            if (!currentBounds) return INITIAL_DATA.allData;
            return INITIAL_DATA.allData.filter(item =>
                item.LATf >= currentBounds.south &&
                item.LATf <= currentBounds.north &&
                item.LONGf >= currentBounds.west &&
                item.LONGf <= currentBounds.east
            );
        }

        function calculateCorrelation(x, y) {
            const n = x.length;
            let sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0, sum_y2 = 0;
            
            for (let i = 0; i < n; i++) {
                sum_x += x[i];
                sum_y += y[i];
                sum_xy += x[i] * y[i];
                sum_x2 += x[i] * x[i];
                sum_y2 += y[i] * y[i];
            }
            
            const numerator = n * sum_xy - sum_x * sum_y;
            const denominator = Math.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));
            
            return denominator === 0 ? 0 : numerator / denominator;
        }

        function updateCorrelationChart(visibleData) {
            if (visibleData.length === 0) {
                Plotly.react('correlation', [{
                    type: 'heatmap',
                    z: [[]],
                    x: [],
                    y: [],
                    colorscale: 'Viridis'
                }], {
                    title: 'Geochemical Correlation (0 amostras)',
                    width: 600,
                    height: 600
                });
                return;
            }

            const correlationColumns = ['SiO2', 'MgO', 'TiO2', 'Al2O3', 'Fe2O3t', 'MnO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'Ba', 'Rb', 'Sr', 'Itrio', 'Zr', 'Ti'];
            const zData = correlationColumns.map(col1 => 
                correlationColumns.map(col2 => {
                    const values1 = visibleData.map(item => parseFloat(item[col1])).filter(v => !isNaN(v));
                    const values2 = visibleData.map(item => parseFloat(item[col2])).filter(v => !isNaN(v));
                    return calculateCorrelation(values1, values2);
                })
            );

            Plotly.react('correlation', [{
                type: 'heatmap',
                z: zData,
                x: correlationColumns,
                y: correlationColumns,
                colorscale: 'Viridis',
                zmin: -1,
                zmax: 1
            }], {
                title: `Geochemical Correlation (${visibleData.length} amostras)`,
                xaxis: { 
                    tickangle: -45,
                    side: 'bottom'
                },
                yaxis: {
                    autorange: 'reversed'
                },
                width: 800,
                height: 800
            });
        }

        function standardize(data) {
            const mean = data.reduce((a, b) => a + b, 0) / data.length;
            const std = Math.sqrt(data.reduce((a, b) => a + (b - mean) ** 2, 0) / data.length);
            return data.map(x => (x - mean) / (std || 1));
        }

        function updateSecondaryCharts() {
            const visibleData = getVisibleData();
            const counts = {};
            visibleData.forEach(item => {
                const cls = item.Classificacao.toLowerCase();
                counts[cls] = (counts[cls] || 0) + 1;
            });

            const labels = Object.keys(counts);
            const values = Object.values(counts);

            // Atualizar gráfico de rosca
            Plotly.react('donut', [{
                type: 'pie',
                labels: labels,
                values: values,
                hole: 0.4,
                marker: { colors: labels.map(l => INITIAL_DATA.colorMap[l]) }
            }], {
                title: `Distribution (${visibleData.length} amostras)`,
                showlegend: true
            });

            // Atualizar gráfico de barras
            Plotly.react('bar', [{
                type: 'bar',
                x: labels,
                y: values,
                marker: { color: labels.map(l => INITIAL_DATA.colorMap[l]) },
                text: values,
                textposition: 'outside'
            }], {
                title: 'Count by magma type',
            });

            // Atualizar apenas a matriz de correlação
            if (visibleData.length >= 2) {
                updateCorrelationChart(visibleData);
            }
        }

        function initDashboard() {
            const mapLayout = INITIAL_DATA.mapData.layout;
            mapLayout.legend = {
                title: { text: 'Magma types' },
                itemsizing: 'constant'
            };

            // Inicializar todos os gráficos
            Plotly.newPlot('map', INITIAL_DATA.mapData.data, mapLayout);
            Plotly.newPlot('donut', INITIAL_DATA.donutData.data, INITIAL_DATA.donutData.layout);
            Plotly.newPlot('bar', INITIAL_DATA.barData.data, INITIAL_DATA.barData.layout);
            Plotly.newPlot('correlation', INITIAL_DATA.correlationData.data, INITIAL_DATA.correlationData.layout);
            Plotly.newPlot('pca', INITIAL_DATA.pcaData.data, INITIAL_DATA.pcaData.layout);

            // Adicionar listeners para interatividade
            document.getElementById('map').on('plotly_relayout', function() {
                currentBounds = getCurrentBounds();
                updateSecondaryCharts();
            });

            document.getElementById('map').on('plotly_restyle', function() {
                updateSecondaryCharts();
            });

            // Listener para redimensionamento da janela
            window.addEventListener('resize', function() {
                const graphs = document.querySelectorAll('.plotly-graph');
                graphs.forEach(graph => {
                    Plotly.Plots.resize(graph);
                });
            });
        }

        // Inicializar o dashboard quando o DOM estiver carregado
        document.addEventListener('DOMContentLoaded', initDashboard);
    </script>
</body>
</html>
"""

def create_pca_plot(df, numeric_columns):
    """Criar gráfico PCA estático"""
    try:
        print("Iniciando criação do PCA...")
        print(f"Colunas numéricas disponíveis: {numeric_columns}")
        
        # Criar cópia para evitar warnings
        df = df.copy()
        
        # Garantir que a classificação esteja em minúsculo
        df['Classificacao'] = df['Classificacao'].str.lower()
        
        # Selecionar apenas colunas numéricas existentes
        numeric_columns = [col for col in numeric_columns if col in df.columns]
        print(f"Colunas após filtro: {numeric_columns}")
        
        if len(numeric_columns) < 2:
            raise ValueError("Número insuficiente de colunas numéricas para PCA")
        
        # Verificar valores ausentes
        print("Verificando valores ausentes...")
        for col in numeric_columns:
            missing = df[col].isna().sum()
            print(f"Coluna {col}: {missing} valores ausentes")
        
        # Preencher valores ausentes
        X = df[numeric_columns].fillna(df[numeric_columns].mean())
        
        # Padronização
        print("Realizando padronização dos dados...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA
        print("Aplicando PCA...")
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_scaled)
        
        # Calcular variância explicada
        var_explicada = pca.explained_variance_ratio_ * 100
        print(f"Variância explicada - PC1: {var_explicada[0]:.1f}%, PC2: {var_explicada[1]:.1f}%")
        
        # Criar DataFrame com resultados
        pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
        pca_df['Classificacao'] = df['Classificacao']
        
        # Criar gráfico base
        print("Criando gráfico...")
        fig = go.Figure()
        
        # Adicionar pontos para cada classificação
        for classificacao in pca_df['Classificacao'].unique():
            mask = pca_df['Classificacao'] == classificacao
            fig.add_trace(go.Scatter(
                x=pca_df.loc[mask, 'PC1'],
                y=pca_df.loc[mask, 'PC2'],
                mode='markers',
                name=classificacao,
                marker=dict(
                    color=colors.get(classificacao, '#CCCCCC'),
                    size=8
                )
            ))
        
        # Adicionar vetores dos componentes
        print("Adicionando vetores dos componentes...")
        for i, feature in enumerate(numeric_columns):
            fig.add_shape(
                type='line',
                x0=0, y0=0,
                x1=pca.components_[0, i] * 3,
                y1=pca.components_[1, i] * 3,
                line=dict(color='black', width=1)
            )
            fig.add_annotation(
                x=pca.components_[0, i] * 3.2,
                y=pca.components_[1, i] * 3.2,
                text=feature,
                showarrow=False,
                font=dict(size=10)
            )
        
        # Atualizar layout
        fig.update_layout(
            title=f'Geostatistical Principal Component Analysis (PC1: {var_explicada[0]:.1f}%, PC2: {var_explicada[1]:.1f}%)',
            xaxis_title='PC1',
            yaxis_title='PC2',
            showlegend=True,
            legend_title_text='Magma types',
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=600,
            height=600
        )
        
        print("PCA criado com sucesso!")
        return fig
    except Exception as e:
        print(f"Erro detalhado ao criar gráfico PCA: {str(e)}")
        print("Tipo do erro:", type(e).__name__)
        import traceback
        print("Traceback completo:")
        print(traceback.format_exc())
        
        fig = go.Figure()
        fig.add_annotation(
            text=f"Erro ao carregar PCA: {str(e)}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False
        )
        return fig

def create_dashboard(amostras_df, base_path):
    # Verificar se a coluna 'Classificacao' está no DataFrame
    if 'Classificacao' not in amostras_df.columns:
        print("Coluna 'Classificacao' não encontrada!")
        return
    
    # Lista específica de colunas para correlação
    COLUNAS_CORRELACAO = ['SiO2', 'MgO', 'TiO2', 'Al2O3', 'Fe2O3t', 'MnO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'Ba', 'Rb', 'Sr', 'Itrio', 'Zr', 'Ti']
    
    # Verificar e filtrar colunas de correlação que existem no DataFrame
    colunas_disponiveis = [col for col in COLUNAS_CORRELACAO if col in amostras_df.columns]
    print(f"Colunas usadas para correlação: {colunas_disponiveis}")
    
    # Garantir que todas classificações tenham cores e estejam em minúsculo
    amostras_df['Classificacao'] = amostras_df['Classificacao'].str.lower()
    
    # Criar mapa com cores explícitas
    fig_map = px.scatter_map(
        amostras_df,
        lat="LATf",
        lon="LONGf",
        color="Classificacao",
        color_discrete_map=colors,
        hover_name="Classificacao",
        zoom=4,
        title="Geographic Distribution",
        labels={"Classificacao": "Magma types"}  # Adicione esta linha
    )
    fig_map.update_layout(
        mapbox_style="carto-positron",
        height=600,
        legend_title_text="Magma types" 
    )

    # Donut (rosca)
    count = amostras_df['Classificacao'].value_counts()
    fig_donut = go.Figure(data=[go.Pie(
        labels=count.index,
        values=count.values,
        hole=0.4,
        marker=dict(colors=[colors.get(cls, '#CCCCCC') for cls in count.index])
    )])
    fig_donut.update_layout(title="Distribution by Classification")
    
    # Barra
    fig_bar = go.Figure(data=[go.Bar(
        x=count.index,
        y=count.values,
        marker=dict(color=[colors.get(cls, '#CCCCCC') for cls in count.index]),
        text=count.values,
        textposition='outside'
    )])
    fig_bar.update_layout(
        title="Count by Classification",
        xaxis=dict(title='Classificação'),
        yaxis=dict(title='Quantidade')
    )
    
    # PCA estático
    fig_pca = create_pca_plot(amostras_df, colunas_disponiveis)
    
    # Correlação inicial com dimensões ajustadas
    corr_matrix = amostras_df[colunas_disponiveis].corr().round(2)
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=colunas_disponiveis,
        y=colunas_disponiveis,
        colorscale='Viridis',
        zmin=-1,
        zmax=1
    ))
    fig_corr.update_layout(
        title='Geochemical Correlation',
        xaxis=dict(
            tickangle=-45,
            side='bottom'
        ),
        yaxis=dict(
            autorange='reversed'
        ),
        width=600,
        height=600
    )

    # Atualizar o template JavaScript
    DASHBOARD_TEMPLATE_UPDATED = Template(DASHBOARD_TEMPLATE).render(
        map_json=fig_map.to_json(),
        donut_json=fig_donut.to_json(),
        bar_json=fig_bar.to_json(),
        correlation_json=fig_corr.to_json(),
        pca_json=fig_pca.to_json(),
        all_data=json.dumps(amostras_df.to_dict(orient='records')),
        colors=colors,
        numeric_columns=colunas_disponiveis
    )
    
    output_path = os.path.join(base_path, "dashboard_final.html")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(DASHBOARD_TEMPLATE_UPDATED)

    print(f"Dashboard criado com sucesso em: {output_path}")

    # Mostra no Streamlit
    with open(output_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    st.components.v1.html(html_content, height=1000, scrolling=True)
  

  

if __name__ == '__main__':
    # Certifique-se que amostras_df e base_path estão definidos
    create_dashboard(amostras_df, base_path) 

