import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import pickle

class Plot():
  def __init__(self, dataset_file, directory='plots/miniw/'):
    self.directory = directory
    os.makedirs(os.path.dirname(self.directory), exist_ok=True)

    with open(dataset_file, 'rb') as f:
      clean_data = np.array(pickle.load(f))
      bad_pos = clean_data.item()['bad_pos']
      self.cover = clean_data.item()['images']
      self.width = self.height = int(np.sqrt(len(self.cover)))

      indices = np.arange(len(self.cover)).reshape((self.width, self.height))
      self.wall_pos = np.where(np.isin(indices, bad_pos))
    import pdb;pdb.set_trace()

  def plot(self, rep, it, dim, name='plot'):
    nrows = (dim - 1) // 5 + 1
    ncols = min(5, dim)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
      axes =  np.array([axes])
    axes = axes.ravel()

    for eigen_id in range(dim):
      heatmap = rep[:, eigen_id].reshape((self.width, self.height))

      ax = axes[eigen_id]
      im = ax.imshow(heatmap, cmap='jet')

      for x, y in zip(self.wall_pos[0], self.wall_pos[1]):
        ax.add_patch(
          patches.Rectangle(
            (y-0.5, x-0.5), 1, 1, linewidth=0, fill=True, color='black')
          )

      ax.set_xticks([])
      ax.set_yticks([])
      divider = make_axes_locatable(ax)
      cax = divider.append_axes('right', size='7%', pad=0.3)
      cbar = plt.colorbar(im, cax=cax)
      cbar.formatter.set_powerlimits((0, 0))
    for j in range(eigen_id+1, len(axes)):
      axes[j].axis('off')
    plt.savefig(f'{self.directory}/iter_{name}_{it}.png')
    plt.close()

  def plot_pos(self, all_agent_pos, it, name='pos'):
    num_pos = len(all_agent_pos)
    fig, axes = plt.subplots(1 * num_pos, 2, figsize=(8, 4 * num_pos))
    axes = axes.ravel()
    ax_id = 0

    for agent_pos in all_agent_pos:
      ax = axes[ax_id]
      im = ax.imshow(agent_pos, cmap='hot', interpolation='nearest')
      ax.set_xticks([])
      ax.set_yticks([])
      divider = make_axes_locatable(ax)
      cax = divider.append_axes('right', size='7%', pad=0.3)
      cbar = plt.colorbar(im, cax=cax)
      cbar.formatter.set_powerlimits((0, 0))

      ax_id +=1
      ax = axes[ax_id]
      im = ax.imshow(np.log(agent_pos+1), cmap='hot', interpolation='nearest')
      ax.set_xticks([])
      ax.set_yticks([])
      divider = make_axes_locatable(ax)
      cax = divider.append_axes('right', size='7%', pad=0.3)
      cbar = plt.colorbar(im, cax=cax)
      cbar.formatter.set_powerlimits((0, 0))
      ax_id +=1

    plt.savefig(f'{self.directory}/iter_{name}_{it}.png')
    plt.close()

