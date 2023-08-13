
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os

class Plot():
  def __init__(self,env, dummy_obs, directory ='plots/minigrid/'):
    self.directory = directory
    os.makedirs(os.path.dirname(self.directory), exist_ok=True)

    self.width, self.height = env.width, env.height
    obs = dummy_obs.observation.copy()
    self.wall_pos = np.where(obs==2)

    default_obs = obs.copy()
    obs[np.where(obs==10)] = 0
    cover = []
    for x in range(self.width):
      for y in range(self.height):
        previous_val = obs[x, y, 0]
        obs[x, y, 0] = 10
        if previous_val == 2 or previous_val == 1:
          cover.append(default_obs)
        else:
          cover.append(obs.copy())
        obs[x, y, 0] = previous_val
    self.cover = np.array(cover)

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
      # divider = make_axes_locatable(ax)
      # cax = divider.append_axes('right', size='7%', pad=0.3)
      # cbar = plt.colorbar(im, cax=cax)
      # cbar.formatter.set_powerlimits((0, 0))
    for j in range(eigen_id+1, len(axes)):
      axes[j].axis('off')
    # plt.tight_layout()
    plt.savefig(f'{self.directory}/iter_{it}_{name}.png', dpi=50)
    plt.close()


  def plot_actions(self, option_actions, it, rep=None, name='action_plot'):
    num_options = len(option_actions)
    nrows = (num_options - 1) // 5 + 1
    ncols = min(5, num_options)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
      axes =  np.array([axes])
    axes = axes.ravel()

    for option_id, actions in enumerate(option_actions):
      actions = actions.reshape((self.width, self.height))
      ax = axes[option_id]
      if rep is not None:
        heatmap = rep[:, option_id].reshape((self.width, self.height))
        im = ax.imshow(heatmap, cmap='jet')
      else:
        im = ax.imshow(actions, cmap='spring')

      # create a grid of coordinates to plot the arrows
      x, y = np.meshgrid(
        np.arange(actions.shape[1]), np.arange(actions.shape[0]))
      u = np.where(actions == 0, 1, np.where(actions == 2, -1, 0))
      v = np.where(actions == 1, -1, np.where(actions == 3, 1, 0))

      # plot the arrows using quiver
      ax.quiver(x, y, u, v, color='black', pivot='mid', scale=30, width=0.007)

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

    for j in range(option_id+1, len(axes)):
      axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(f'{self.directory}/iter_{it}_{name}.png', dpi=50)
    plt.close()


  def plot_pos(self, counts, it, name='pos'):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes = axes.ravel()
    ax_id = 0

    agent_pos = np.zeros((self.width, self.height))
    for key, value in counts.items():
      agent_pos[key] = value

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

    plt.savefig(f'{self.directory}/iter_{it}_{name}.png')
    plt.close()

