"""python app.py로도 동일한 웹캠 앱을 실행한다."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).with_name('05_app.py')), run_name='__main__')
