import os
import requests
from time import sleep

FONTS = {
    # Indic
    'NotoSans-Regular.ttf': 'https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf',
    'NotoSansDevanagari-Regular.ttf': 'https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf',
    'NotoSansBengali-Regular.ttf': 'https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSansBengali/NotoSansBengali-Regular.ttf',
    'NotoSansGujarati-Regular.ttf': 'https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSansGujarati/NotoSansGujarati-Regular.ttf',
    'NotoSansGurmukhi-Regular.ttf': 'https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSansGurmukhi/NotoSansGurmukhi-Regular.ttf',
    'NotoSansTamil-Regular.ttf': 'https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSansTamil/NotoSansTamil-Regular.ttf',
    'NotoSansTelugu-Regular.ttf': 'https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSansTelugu/NotoSansTelugu-Regular.ttf',
    'NotoSansKannada-Regular.ttf': 'https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSansKannada/NotoSansKannada-Regular.ttf',
    'NotoSansMalayalam-Regular.ttf': 'https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSansMalayalam/NotoSansMalayalam-Regular.ttf',
    'NotoSansOdia-Regular.ttf': 'https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSansOdia/NotoSansOdia-Regular.ttf',
    # Arabic
    'NotoNaskhArabic-Regular.ttf': 'https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoNaskhArabic/NotoNaskhArabic-Regular.ttf',
    # CJK (large, optional)
    'NotoSansSC-Regular.otf': 'https://raw.githubusercontent.com/googlefonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansSC-Regular.otf',
    'NotoSansJP-Regular.otf': 'https://raw.githubusercontent.com/googlefonts/noto-cjk/main/Sans/OTF/Japanese/NotoSansJP-Regular.otf',
    'NotoSansKR-Regular.otf': 'https://raw.githubusercontent.com/googlefonts/noto-cjk/main/Sans/OTF/Korean/NotoSansKR-Regular.otf',
}


def download_fonts(target_dir='fonts'):
    os.makedirs(target_dir, exist_ok=True)
    for name, url in FONTS.items():
        dest = os.path.join(target_dir, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            print(f'Skipping (exists): {name}')
            continue
        print(f'Downloading {name}...')
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200 and resp.content:
                with open(dest, 'wb') as f:
                    f.write(resp.content)
                print('Saved', dest)
            else:
                print('Failed to download', name, 'status', resp.status_code)
        except Exception as e:
            print('Error downloading', name, e)
        sleep(0.5)


if __name__ == '__main__':
    print('Downloading fonts into ./fonts (this may take a while for CJK fonts)')
    download_fonts()
    print('Done. Restart your app and ensure REPORTLAB_FONT_PATH points to the ./fonts folder or let app detect fonts automatically.')
