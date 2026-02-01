import sys, os, pickle
import numpy as np
import torch
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from plotnine import *
from tensorboard.backend.event_processing.event_accumulator \
    import EventAccumulator

import deep_spine.utils.utils as utils
from plotnine.data import mtcars
import seaborn as sns

from scipy.io import savemat
from tbparse import SummaryReader

def log_2_df(logdir):
    reader = SummaryReader(logdir, pivot=False)
    df = reader.scalars
    return df

def df_to_mat_table(df, filename):
    mat_dict = {col: df[col].to_numpy() for col in df.columns}
    savemat(filename, {"table": mat_dict})


def training_logs_to_dataframe(
        path_list,
        savefile=None
    ):
    logpath_list = [os.path.abspath(f'{p}/tb') for p in path_list]
    agentpath_list = [os.path.abspath(f'{p}') for p in path_list]
    all_data = []
    for l in logpath_list:
        for f in os.listdir(l):
            subdir_path = os.path.join(l, f)
            if os.path.isdir(subdir_path):
                for sf in os.listdir(subdir_path):
                    subsubdir_path = os.path.join(subdir_path, sf)
                    if os.path.isdir(subsubdir_path):
                        events = (
                            [os.path.join(subsubdir_path, f)
                            for f in os.listdir(subsubdir_path)
                                if 'tfevents' in f]
                        )
                    for event in events:
                        event_acc = EventAccumulator(event)
                        event_acc.Reload()
                        if 'scalars' in event_acc.Tags():
                            scalar_tags = event_acc.Tags()['scalars']
                            for tag in scalar_tags:
                                data = event_acc.Scalars(tag)
                                df = pd.DataFrame(
                                    data=[(s.step, s.value) for s in data],
                                    columns=['Step', 'Value']
                                )
                                model_str = os.path.basename(subdir_path)
                                model = model_str.split('-')

                                df['Tag'] = tag
                                df['ModelID'] = model_str
                                df['Model'] = '-'.join(model[1:])
                                all_data.append(df)

    df = pd.concat(all_data, ignore_index=True)
    tags = df['Tag'].unique()
    for tag in tags:
         if 'eps/' in tag:
             df = insert_max_value_by_tag(df,tag)
             df = insert_2norm_value_by_tag(df,tag)
             #df = insert_checkpoint_values_and_save_performance(
             #   df, agentpath_list
             #)
    if savefile:
        df.to_pickle(savefile)
    return df


def insert_max_value_by_tag(df, tag_value):
    filtered_df = df[df['Tag'] == tag_value]
    idx = filtered_df.groupby('ModelID')['Value'].idxmax()
    max_rows = filtered_df.loc[idx]
    max_rows['Tag'] = 'max-' + tag_value
    new_df = pd.concat([df, max_rows], ignore_index=True)
    return new_df

def insert_2norm_value_by_tag(df, tag_value):
    def calc_2norm(series):
        return np.sqrt((series**2).sum())
    filtered_df = df[df['Tag'] == tag_value]
    norm = filtered_df.groupby('ModelID')['Value'].transform(calc_2norm)
    norm_df = filtered_df.drop_duplicates(subset='ModelID').copy()
    norm_df['Value'] = norm.drop_duplicates()
    norm_df['Tag'] = 'norm-' + tag_value
    new_df = pd.concat([df, norm_df], ignore_index=True)
    return new_df

def insert_checkpoint_values_and_save_performance(df,agentpath_list):
  all_data = []
  for path in agentpath_list:
    agent_performance_dict = {}
    for a in os.listdir(path):
      if a.endswith('.pkl'):
        continue
      zips = utils.get_all_agent_checkpoints(os.path.join(path, a, "model"))
      checkpoints = [float(name.split('.')[0]) for name in zips]
      unique_agents = df['ModelID'].unique()
      agentid = next((qid for qid in unique_agents if str(a) in qid), None)
      agent_performance = np.zeros((len(checkpoints),2))
      if agentid is not None:
        agent_df = (
            df[(df['ModelID'] == agentid)
               & (df['Tag'] == 'eps/eps')]
        )
        for i, checkpoint in enumerate(checkpoints):
          print(i, checkpoint)
          match = agent_df[agent_df['Step'] == int(checkpoint)]
          print(sorted(agent_df['Step'].unique()[:20]))
          print(match)
          model = (
              agent_df.loc[agent_df['Step'] == int(checkpoint),'Model'].iloc[0]
          )
          value = match['Value'].values[0]
          all_data.append({
            'ModelID': agentid,
            'Model': model,
            'Step': int(checkpoint),
            'Value': value,
            'Tag': 'Checkpoints'
          })
          agent_performance[i] = [checkpoint, value]

        sorted_indices = np.argsort(agent_performance[:,1])
        sorted_indices = sorted_indices[::-1]
        agent_performance = agent_performance[sorted_indices]
        agent_performance_dict[a] = [model, agent_performance]

    pickle.dump(agent_performance_dict,
                open(path + '/agent_performance_dict.pkl','wb'))

  checkpoints_df = pd.DataFrame(all_data)
  return pd.concat([df, checkpoints_df], ignore_index=True)


def plot_df_by_step(df,value_string):
  filtered_df = df[df['Tag'] == value_string]

  max_df = df[df['Tag'] == 'max-' + value_string]
  plot = (ggplot(aes(x='Step', y='Value', color='Model', group='ModelID'))
          + geom_line(data=filtered_df, alpha=0.5)
          + geom_point(data=max_df, alpha=0.5,shape='x')
          + ylab(f'{value_string}')
          + xlab('training steps')
          + theme_bw()
  )

  if value_string == 'eps/ep_len_sec':
    cp_df = df[df['Tag'] == 'Checkpoints']
    plot += geom_point(data=cp_df, alpha=0.5)

  plot.show()

def plot_df_boxplot(df,value_string):

    filtered_df = df.loc[df['Tag'] == value_string].copy()

    # Create a color map for each unique Model
    unique_models = filtered_df['Model'].unique()
    palette = sns.color_palette('husl', len(unique_models))
    color_map = {model: palette[i] for i, model in enumerate(unique_models)}
    # Assign colors to models
    filtered_df['Color'] = filtered_df['Model'].map(color_map)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(20, 8))

    # Create the boxplot
    box_plot = sns.boxplot(x='Model', y='Value', data=filtered_df,
                           hue='Model',
                           width=0.5,
                           fill=False,
                           gap=0.1,
                           palette=color_map,
                           ax=ax, dodge=False)


    # Add jittered points with different colors for each Model
    for model in unique_models:
        sns.stripplot(x='Model', y='Value',
                      data=filtered_df[filtered_df['Model'] == model],
                      ax=ax, color=color_map[model], size=5,
                      jitter=True, alpha=0.5, label=model)

    # Add the mean points with different colors for each Model
    means = filtered_df.groupby('Model')['Value'].mean().reset_index()
    means['Color'] = means['Model'].map(lambda m: filtered_df[filtered_df['Model'] == m]['Color'].iloc[0])
    for model, color in zip(means['Model'], means['Color']):
        sns.scatterplot(x=[model], y=means[means['Model'] == model]['Value'], ax=ax, color=color, s=100, marker='x', zorder=10)

    # Set labels and title
    ax.set_ylabel(value_string)
    ax.set_xlabel('')

    # Remove x-axis text
    ax.set_xticklabels([])

    # Add the legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[label], markersize=10) for label in unique_models]
    ax.legend(handles, unique_models, title='Model',
        bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small'
    )

    # Adjust layout to fit everything
    plt.subplots_adjust(left=0.05, right=0.4)  # Make the plot area narrower
    plt.show()

def plot_arm26_trial(
        filename,
        start_index: int = 0,
        end_index = None,
    ):

    def ax_style(ax):
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([])
        ax.set_xlabel('')

    d = np.loadtxt(filename, delimiter=',')
    print(d.shape)

    pos = d[:,:2]
    vel = d[:,2:4]
    sensory = d[:,4:16]
    posd = d[:,16:18]
    noise = d[:,18:20]
    motor = d[:,20:]

    print(f'mean pos = {np.rad2deg(np.mean(pos[-100:,:], axis=0))} degrees')
    print(f'mean pos = {np.mean(pos[-100:,:], axis=0)} radians')
    return

    effort = np.linalg.norm(motor[start_index:end_index,:])
    print(f'\neffort =  {effort}\n')
    error = (
        np.linalg.norm(
            posd[start_index:end_index,:] - pos[start_index:end_index,:]
        )
    )
    print(f'\nerror =  {error}\n')

    win = [start_index, end_index]
    plt.rcParams['lines.linewidth'] = 0.5
    fig = plt.figure(figsize=(15,8))


    outer_grid = gridspec.GridSpec(5, 1, height_ratios=[1, 1, 1, 1, 1])

    ax = plt.Subplot(fig, outer_grid[0])
    fig.add_subplot(ax)
    ax.plot(np.linalg.norm(posd-pos,axis=1))
    if end_index:
        plt.xlim(win[0],win[1])
    plt.ylabel('error')
    ax_style(ax)


    ax = plt.Subplot(fig, outer_grid[1])
    fig.add_subplot(ax)
    ax.plot(noise)
    if end_index:
        plt.xlim(win[0],win[1])
    plt.ylabel('noise')
    ax_style(ax)

    ax = plt.Subplot(fig, outer_grid[2])
    fig.add_subplot(ax)
    ax.plot(sensory[:,:6])
    if end_index:
        plt.xlim(win[0],win[1])
    plt.ylabel('length')
    ax_style(ax)

    ax = plt.Subplot(fig, outer_grid[3])
    fig.add_subplot(ax)
    ax.plot(sensory[:,6:])
    if end_index:
        plt.xlim(win[0],win[1])
    plt.ylabel('velocity')
    ax_style(ax)

    inner_grid = gridspec.GridSpecFromSubplotSpec(
            4, 1, subplot_spec=outer_grid[4]
    )

    ax = plt.Subplot(fig, inner_grid[0, 0])
    fig.add_subplot(ax)
    ax.plot(motor[:,0:2])
    plt.ylim([0,1.2])
    if end_index:
        plt.xlim(win[0],win[1])
    ax_style(ax)

    ax = plt.Subplot(fig, inner_grid[1, 0])
    fig.add_subplot(ax)
    ax.plot(motor[:,2:4])
    plt.ylim([0,1.2])
    if end_index:
        plt.xlim(win[0],win[1])
    ax_style(ax)

    ax = plt.Subplot(fig, inner_grid[2, 0])
    fig.add_subplot(ax)
    ax.plot(motor[:,4:6])
    plt.ylim([0,1.2])
    if end_index:
        plt.xlim(win[0],win[1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle(f'{filename}            ||u|| = {str(round(effort,2))}'
        + f', ||e|| = {str(round(error,2))}'
    )

    fig.tight_layout()
    plt.show()


def linkx():
    axes = plt.gcf().axes
    parent = axes[0]
    for i in range(1,len(axes)):
        axes[i].sharex(parent)
    for ax in axes[:-1]:
        ax.tick_params(bottom=False, labelbottom=False)

def plot_vertical_stack(
    fig = None,
    figsize = (15,8),
    data = None,
    data_str = None,
    start_index = None,
    end_index = None,
    ylim = None
    ):

    def ax_style(ax):
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([])
        ax.set_xlabel('')

    n = len(data)
    win = [start_index, end_index]
    if fig is None:
        plt.rcParams['lines.linewidth'] = 0.5
        fig = plt.figure(figsize=figsize)

    outer_grid = gridspec.GridSpec(
        n, 1, height_ratios=[1] * n
    )
    dt = 1/200
    N = data[0].shape[0]
    t = np.arange(0, N * dt, dt)

    for i, data in enumerate(data):
        ax = plt.Subplot(fig, outer_grid[i])
        fig.add_subplot(ax)
        ax.plot(t,data)
        ax.axhline(y = 0, color = 'k', linestyle = '-') 
        if end_index:
            plt.xlim(win[0],win[1])
        if ylim:
            if len(ylim) != 1:
                plt.ylim(ylim[i][0],ylim[i][1])
            else:
                plt.ylim(ylim[0][0],ylim[0][1])
        plt.ylabel(data_str[i])
        ax_style(ax)

    linkx()

    plt.xlabel('time (s)')
    num_digits = len(str(int(t[-1])))
    scale = 10 ** (num_digits - 1)
    ax.set_xticks(np.arange(0, t[-1] + scale, scale))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)


    fig.tight_layout()
    plt.show()
    return fig


def plot_heb_trial(
        filename,
        start_index: int = 0,
        end_index = None,
    ):

    def ax_style(ax):
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([])
        ax.set_xlabel('')

    d = np.loadtxt(filename, delimiter=',')
    N = d.shape[0]
    dt = 1/200
    t = np.arange(0, N * dt, dt)
    p_pre = d[:,:6]
    p_post = d[:,6:12]
    hp_p_post = d[:,12:18]
    mean_p_post = d[:,18:24]
    outer_product = d[:,24:60]
    weighted_outer_product = d[:,60:96]
    learning_signal = d[:,96:132]
    compensation = d[:,132:168]
    rate = d[:,168:204]

    plot_vertical_stack(
        data = [p_pre, p_post, hp_p_post, mean_p_post,
        ],
        data_str = ['p_pre', 'p_post', 'hp_p_post', 'mean_p_post',
        ],
        start_index = start_index,
        end_index = end_index
    )
    plot_vertical_stack(
        data = [learning_signal, compensation, rate, 
        ],
        data_str = ['outer_product', 'weighted_outer_product',
                    'learning_signal', 'compensation', 'rate'
        ],
        start_index = start_index,
        end_index = end_index
    )


def plot_arm26babble_trial(
        filename,
        start_index: int = 0,
        end_index = None,
    ):

    def ax_style(ax):
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([])
        ax.set_xlabel('')

    d = np.loadtxt(filename, delimiter=',')
    time = d[:,0]
    qpos = d[:,1:3]
    qvel = d[:,3:5]
    length = d[:,5:11]
    velocity = d[:,11:17]
    activation_II = d[:,17:23]
    activation_Ia = d[:,23:29]
    activation_MN = d[:,29:35]
    activation_muscle = d[:,35:41]
    descending_drive = d[:,41:47]

    plot_vertical_stack(
        data = [qpos, qvel],
        data_str = ['qpos', 'qvel'],
        start_index = start_index,
        end_index = end_index
    )

    plot_vertical_stack(
        data = [length, velocity,
                activation_II, activation_Ia,
                activation_MN, activation_muscle,
                descending_drive
        ],
        data_str = ['length', 'velocity',
                    'II', 'Ia',
                    'MN', 'muscle activation',
                    'babble'
        ],
        start_index = start_index,
        end_index = end_index,
        ylim = [[0,1],[-2,2],[0,1],[0,1],[0,1],[0,1],[0,1]]
    )


def plot_df_performance_index(df,value_string):
  eps_df = df[df['Tag'] == 'eps/eps'].reset_index(drop=True)
  value_df = df[df['Tag'] == value_string].reset_index(drop=True)
  plot_df = pd.DataFrame({
      'Eps' : eps_df['Value'],
      'Value' : value_df['Value'],
      'Model' : eps_df['Model'],
      'ModelID' : eps_df['ModelID']
  })


  idx = plot_df.groupby('ModelID')['Value'].idxmax()
  result_df = plot_df.loc[idx]
  result_df['Value'] = result_df['Value'] / result_df['Eps']
  result_df = result_df.reset_index(drop=True)

  plot = (ggplot(result_df,aes(x='Model',
                                 y='Value',
                                 color='Model'))
          + geom_boxplot()
          + stat_summary(aes(group='Model',
                             color='Model'),
                         fun_y=np.mean,
                         geom='point',
                         size=3,
                         shape='x')
          + geom_jitter(width=0.2, size=2, alpha=0.5)
          + ylab(f'{value_string}')
          + theme_bw()
          + theme(figure_size=(3, 5),
                  axis_text_x=element_blank())
          )
  #display(plot)

def plot_weights(W, fig=None, ax=None, im=None, save_dir="figures",
                 title='Weights Matrix', mn_labels=None, ia_labels=None,
                 scale_opt=None):

    if mn_labels is None:
        mn_labels = ['MN_SF', 'MN_SE', 'MN_EF', 'MN_EE', 'MN_BF', 'MN_BE']
    if ia_labels is None:
        ia_labels = ['Ia_SF', 'Ia_SE', 'Ia_EF', 'Ia_EE', 'Ia_BF', 'Ia_BE']
    W = W.detach().cpu().numpy()
    plt.ion()  # Turn on interactive mode
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

        if scale_opt:
            im = sns.heatmap(
                W, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, vmin=-scale_opt,
                vmax=scale_opt,
                yticklabels=mn_labels, xticklabels=ia_labels, ax=ax)
        else:
            im = sns.heatmap(W, annot=True, fmt='.2f',
                cmap='coolwarm', vmin=0, vmax=1,
                yticklabels=mn_labels,
                xticklabels=ia_labels, ax=ax)


        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.set_title(title, pad=-50)
    else:
        # Remove the colorbar manually if it exists
        if ax.collections:
            for cbar in ax.collections:
                if cbar.colorbar:
                    cbar.colorbar.remove()
        # Clear the previous heatmap
        ax.clear()

        if scale_opt:
            im = sns.heatmap(
                W, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, vmin=-scale_opt,
                vmax=scale_opt,
                yticklabels=mn_labels,
                xticklabels=ia_labels, ax=ax, cbar=True)
        else:
            im = sns.heatmap(W, annot=True, fmt='.2f',
                         cmap='coolwarm', vmin=0, vmax=1,
                         yticklabels=mn_labels,
                         xticklabels=ia_labels,
                         ax=ax, cbar=True)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.set_title(title, pad=-50)
    plt.draw()
    plt.pause(0.05)  # Pause to render the plot
    plt.show()
    plt.ioff()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        files = os.listdir(save_dir)
        existing_files = [
            f for f in files if f.startswith("plot_") and f.endswith(".png")
        ]
        if existing_files:
            # Extract numeric part from the filenames and find the max
            existing_numbers = [
                int(f.split('_')[1].split('.')[0]) for f in existing_files
            ]
            next_number = max(existing_numbers) + 1
        else:
            next_number = 1

        filename = f"plot_{next_number:09d}.png"
        save_path = os.path.join(save_dir, filename)
        fig.savefig(save_path)
    return fig, ax, im
