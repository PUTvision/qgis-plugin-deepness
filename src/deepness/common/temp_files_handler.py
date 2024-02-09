import os.path as path
import shutil
from tempfile import mkdtemp


class TempFilesHandler:
    def __init__(self) -> None:
        self._temp_dir = mkdtemp()
        
        print(f'Created temp dir: {self._temp_dir} for processing')

    def get_results_img_path(self):
        return path.join(self._temp_dir, 'results.dat')

    def get_area_mask_img_path(self):
        return path.join(self._temp_dir, 'area_mask.dat')

    def __del__(self):
        shutil.rmtree(self._temp_dir)
