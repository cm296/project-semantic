from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid


def view_filts(w, n_cols, save_path=None):
    grid = make_grid(w, nrow=n_cols)
    grid -= grid.min()
    grid /= grid.max()
    grid = to_pil_image(grid)
    if save_path is None:
        grid.show()
    else:
        grid.save(save_path)
