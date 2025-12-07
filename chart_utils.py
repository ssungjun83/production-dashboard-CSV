"""
차트 스타일 유틸리티 모듈
모든 그래프의 일관된 스타일링을 위한 설정과 함수들
"""
import json
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional

# 차트 스타일 설정 로드
def load_chart_styles() -> Dict[str, Any]:
    """차트 스타일 설정을 JSON 파일에서 로드"""
    try:
        with open('chart_styles.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # 기본 설정 반환
        return {
            "chart_config": {"default_height": 600, "text_font_size": 16},
            "colors": {"factory_colors": {"A관": "#1f77b4", "C관": "#aec7e8", "S관": "#ff0000"}},
            "legend_config": {"orientation": "h", "yanchor": "top", "y": 0.98, "xanchor": "right", "x": 0.98}
        }

# 전역 설정
CHART_STYLES = load_chart_styles()

def get_factory_color(factory_name: str) -> str:
    """공장명에 따른 색상 반환"""
    factory_colors = CHART_STYLES["colors"]["factory_colors"]
    for key, color in factory_colors.items():
        if key in factory_name:
            return color
    return CHART_STYLES["colors"]["default_color"]

def get_process_color(process_name: str) -> str:
    """공정명에 따른 색상 반환"""
    process_colors = CHART_STYLES["colors"]["process_colors"]
    return process_colors.get(process_name, CHART_STYLES["colors"]["default_color"])

def apply_standard_layout(fig: go.Figure, 
                         title: str = "",
                         has_text_labels: bool = False,
                         has_legend: bool = True) -> go.Figure:
    """표준 레이아웃을 차트에 적용"""
    
    # 마진 설정
    margin_config = CHART_STYLES["margins"]["with_text_labels"] if has_text_labels else CHART_STYLES["margins"]["default"]
    
    # 범례 설정
    legend_config = CHART_STYLES["legend_config"].copy() if has_legend else None
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=CHART_STYLES["chart_config"]["title_font_size"]),
            x=0.5,
            xanchor='center'
        ),
        height=CHART_STYLES["chart_config"]["default_height"],
        margin=margin_config,
        legend=legend_config,
        showlegend=has_legend
    )
    
    # 축 설정
    axis_config = CHART_STYLES["axis_config"]
    fig.update_xaxes(**axis_config["x_axis"])
    fig.update_yaxes(**axis_config["y_axis"])
    
    return fig

def create_standard_line_chart(df, x: str, y: str, color: str, title: str = "") -> go.Figure:
    """표준 스타일의 라인 차트 생성"""
    line_config = CHART_STYLES["line_chart"]
    text_config = CHART_STYLES["text_config"]
    
    fig = px.line(df, x=x, y=y, color=color, 
                  title=title, markers=True, text=y)
    
    fig.update_traces(
        mode=line_config["mode"],
        marker=dict(size=line_config["marker_size"]),
        line=dict(width=line_config["line_width"]),
        texttemplate=text_config["percentage_format"],
        textposition=text_config["position"],
        textfont=text_config["font"]
    )
    
    return apply_standard_layout(fig, title, has_text_labels=True)

def create_standard_bar_chart(df, x: str, y: str, color: Optional[str] = None, 
                             title: str = "", orientation: str = 'v') -> go.Figure:
    """표준 스타일의 막대 차트 생성"""
    bar_config = CHART_STYLES["bar_chart"]
    
    fig = px.bar(df, x=x, y=y, color=color, 
                 title=title, orientation=orientation)
    
    fig.update_traces(
        textposition=bar_config["textposition"],
        texttemplate=bar_config["texttemplate"]
    )
    
    return apply_standard_layout(fig, title, has_text_labels=True)

def create_combo_chart(df, x_col: str, bar_y: str, line_y: str, 
                      color_col: Optional[str] = None, title: str = "") -> go.Figure:
    """표준 스타일의 콤보 차트 (막대+라인) 생성"""
    from plotly.subplots import make_subplots
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 그룹별 색상 적용
    if color_col:
        for group_name in df[color_col].unique():
            df_group = df[df[color_col] == group_name]
            color = get_factory_color(group_name) if '공장' in color_col else get_process_color(group_name)
            
            # 막대 차트
            fig.add_trace(go.Bar(
                x=df_group[x_col], y=df_group[bar_y],
                name=f'{group_name} {bar_y}',
                marker_color=color,
                text=df_group[bar_y],
                texttemplate=CHART_STYLES["text_config"]["number_format"],
                textposition='outside'
            ), secondary_y=False)
            
            # 라인 차트  
            fig.add_trace(go.Scatter(
                x=df_group[x_col], y=df_group[line_y],
                name=f'{group_name} {line_y}',
                mode='lines+markers+text',
                line=dict(color=color),
                text=df_group[line_y],
                texttemplate=CHART_STYLES["text_config"]["percentage_format"],
                textposition='top center'
            ), secondary_y=True)
    
    return apply_standard_layout(fig, title, has_text_labels=True)

def update_chart_colors_by_factory(fig: go.Figure, df, group_col: str = '공장') -> go.Figure:
    """공장별 색상을 차트에 적용"""
    if group_col in df.columns:
        for i, trace in enumerate(fig.data):
            if hasattr(trace, 'name') and trace.name:
                for factory in df[group_col].unique():
                    if factory in trace.name:
                        color = get_factory_color(factory)
                        trace.update(marker_color=color, line_color=color)
    return fig