import matplotlib.font_manager as fm

# 列出所有已安装字体
font_list = fm.findSystemFonts()
for font_path in font_list:
    try:
        prop = fm.FontProperties(fname=font_path)
        # 只打印可能是中文字体的
        name = prop.get_name()
        if any(key in name.lower() for key in ['hei', 'song', 'sim', 'kai', 'ming', 'cjk', 'noto']):
            print(f"字体名: {name}, 路径: {font_path}")
    except:
        pass