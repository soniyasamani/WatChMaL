"""
Utils for plotting model performance
"""

import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from functools import reduce

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def disp_learn_hist(location, title=None, losslim=None, axis=None, show=True):
    """
    Purpose : Plot the loss and accuracy history for a training session
    
    Args: 
        location    ... output directory containing log files
        title       ... the title for the plot
        losslim     ... sets bound on y axis of loss
        axis        ... axis to plot on
        show        ... if true then display figure, otherwise return figure
    """
    val_log=location + '/log_val.csv'
    val_log_df  = pd.read_csv(val_log)

    train_log_df = get_aggregated_train_data(location)

    if axis is None:
        fig, ax1 = plt.subplots(figsize=(12,8),facecolor='w')
    else:
        ax1 = axis
    
    line11 = ax1.plot(train_log_df.iteration, train_log_df.loss, linewidth=2, label='Train loss', color='b', alpha=0.3)
    line12 = ax1.plot(val_log_df.iteration, val_log_df.loss, marker='o', markersize=3, linestyle='', label='Validation loss', color='blue')

    if losslim is not None:
        ax1.set_ylim(0.,losslim)
    
    ax2 = ax1.twinx()
    line21 = ax2.plot(train_log_df.iteration, train_log_df.accuracy, linewidth=2, label='Train accuracy', color='r', alpha=0.3)
    line22 = ax2.plot(val_log_df.iteration, val_log_df.accuracy, marker='o', markersize=3, linestyle='', label='Validation accuracy', color='red')

    ax1.set_xlabel('Iteration',fontweight='bold',fontsize=24,color='black')
    ax1.tick_params('x',colors='black',labelsize=18)
    ax1.set_ylabel('Loss', fontsize=24, fontweight='bold',color='b')
    ax1.tick_params('y',colors='b',labelsize=18)
    
    ax2.set_ylabel('Accuracy', fontsize=24, fontweight='bold',color='r')
    ax2.tick_params('y',colors='r',labelsize=18)
    ax2.set_ylim(0.,1.05)

    # added these four lines
    lines  = line11 + line12 + line21 + line22
    labels = [l.get_label() for l in lines]
    leg    = ax2.legend(lines, labels, fontsize=16, loc=5, numpoints=1)
    leg_frame = leg.get_frame()
    leg_frame.set_facecolor('white')

    if title is not None:
        ax1.set_title(title, fontsize=20)

    if show:
        plt.grid()
        plt.show()
        return
    
    if axis is None:
        return fig
        
def get_aggregated_train_data(location):
    """
    Aggregate training logs from all processes into a single set of data

    Args:
        location    ... path to outputs directory containing training logs

    Returns: pandas dataframe containing aggregated data
    """
    # get all training data files
    base_log_path = location + '/log_train_[0-9]*.csv'
    log_paths = glob.glob(base_log_path)

    print("Found training logs: ", log_paths)
    
    log_dfs = []
    for log_path in log_paths:
        log_dfs.append(pd.read_csv(log_path))
        log_dfs.append(pd.read_csv(log_path))
    
    # combine all files into one dataframe
    train_log_df = pd.DataFrame(0, index=np.arange(len(log_dfs[0])), columns=log_dfs[0].columns)
    for idx, df_vals in enumerate(zip(*[log_df.values for log_df in log_dfs])):
        iteration = df_vals[0][0]
        epoch = df_vals[0][1]
        loss = sum([df_val[2] for df_val in df_vals]) / len(df_vals)
        accuracy = sum([df_val[3] for df_val in df_vals]) / len(df_vals)

        output_df_vals = (iteration, epoch, loss, accuracy)
        train_log_df.iloc[idx] = output_df_vals

    return train_log_df

def disp_learn_hist_smoothed(location, losslim=None, window_train=400, window_val=40, show=True):
    """
    Plot the loss and accuracy history for a training session with averaging to clean up noise
    
    Args: location      ... output directory containing log files
          losslim       ... sets bound on y axis of loss
          window_train  ... window to average training data over
          window_val    ... window to average validation data over
          show          ... if true then display figure, otherwise return figure
    """
    val_log = location + '/log_val.csv'
    val_log_df   = pd.read_csv(val_log)

    train_log_df = get_aggregated_train_data(location)

    iteration_train    = moving_average(np.array(train_log_df.iteration),window_train)
    accuracy_train = moving_average(np.array(train_log_df.accuracy),window_train)
    loss_train     = moving_average(np.array(train_log_df.loss),window_train)
    
    iteration_val    = moving_average(np.array(val_log_df.iteration),window_val)
    accuracy_val = moving_average(np.array(val_log_df.accuracy),window_val)
    loss_val     = moving_average(np.array(val_log_df.loss),window_val)

    iteration_val_uns    = np.array(val_log_df.iteration)
    accuracy_val_uns = np.array(val_log_df.accuracy)
    loss_val_uns     = np.array(val_log_df.loss)

    saved_best      = np.array(val_log_df.saved_best)
    stored_indices  = np.where(saved_best>1.0e-3)
    iteration_val_st    = iteration_val_uns[stored_indices]
    accuracy_val_st = accuracy_val_uns[stored_indices]
    loss_val_st     = loss_val_uns[stored_indices]

    fig, ax1 = plt.subplots(figsize=(12,8), facecolor='w')
    line11 = ax1.plot(iteration_train, loss_train, linewidth=2, label='Average training loss', color='b', alpha=0.3)
    line12 = ax1.plot(iteration_val, loss_val, label='Average validation loss', color='blue')
    line13 = ax1.scatter(iteration_val_st, loss_val_st, label='BEST validation loss',
                         facecolors='none', edgecolors='blue',marker='o')
    
    ax1.set_xlabel('Iteration',fontweight='bold',fontsize=24,color='black')
    ax1.tick_params('x',colors='black',labelsize=18)
    ax1.set_ylabel('Loss', fontsize=24, fontweight='bold',color='b')
    ax1.tick_params('y',colors='b',labelsize=18)

    if losslim is not None:
        ax1.set_ylim(0.,losslim)
    
    ax2 = ax1.twinx()
    line21 = ax2.plot(iteration_train, accuracy_train, linewidth=2, label='Average training accuracy', color='r', alpha=0.3)
    line22 = ax2.plot(iteration_val, accuracy_val, label='Average validation accuracy', color='red')
    line23 = ax2.scatter(iteration_val_st, accuracy_val_st, label='BEST accuracy',
                         facecolors='none', edgecolors='red',marker='o')
    
    
    ax2.set_ylabel('Accuracy', fontsize=24, fontweight='bold',color='r')
    ax2.tick_params('y',colors='r',labelsize=18)
    ax2.set_ylim(0.,1.0)
    
    # added these four lines
    lines  = line11+ line12+ [line13]+ line21+ line22+ [line23]
    #lines_sctr=[line13,line23]
    #lines=lines_plt+lines_sctr

    labels = [l.get_label() for l in lines]
    
    leg    = ax2.legend(lines, labels, fontsize=16, loc=5, numpoints=1)
    leg_frame = leg.get_frame()
    leg_frame.set_facecolor('white')

    if show:
        plt.grid()
        plt.show()
        return

    return fig

def moving_average(a, n=3) :
    """
    Compute average of a over windows of size n
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_confusion_matrix(labels, predictions, class_names, title = None):
    """
    Plot the confusion matrix for a given energy interval
    
    Args: 
        labels        ... 1D array of true label value, the length = sample size
        predictions   ... 1D array of predictions, the length = sample size
        class_names   ... 1D array of string label for classification targets, the length = number of categories
    """
    fig, ax = plt.subplots(figsize=(12,8),facecolor='w')
    num_labels = len(class_names)
    max_value = np.max([np.max(np.unique(labels)),np.max(np.unique(labels))])

    assert max_value < num_labels

    mat,_,_,im = ax.hist2d(predictions, labels,
                           bins=(num_labels,num_labels),
                           range=((-0.5,num_labels-0.5),(-0.5,num_labels-0.5)),cmap=plt.cm.Blues)

    # Normalize the confusion matrix
    mat = mat.astype("float") / mat.sum(axis=0)#[:, np.newaxis]

    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=20) 
        
    ax.set_xticks(np.arange(num_labels))
    ax.set_yticks(np.arange(num_labels))
    ax.set_xticklabels(class_names,fontsize=20)
    ax.set_yticklabels(class_names,fontsize=20)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax.set_xlabel('Prediction',fontsize=20)
    ax.set_ylabel('True Label',fontsize=20)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(i,j, r"${0:0.3f}$".format(mat[i,j]),
                    ha="center", va="center", fontsize=20,
                    color="white" if mat[i,j] > (0.5*mat.max()) else "black")
    fig.tight_layout()
    plt.title("Confusion matrix: %s"%(title), fontsize=20) 
   
    plt.show()

import matplotlib.pyplot as plt
from itertools import cycle

def plot_classifier_response(softmaxes, labels, particle_names, label_dict,model=None,
                            bins=None, legend_locs=None, title=None,
                            extra_panes=[], xlim=None, label_size=14, legend_label_dict=None, show=True):
    '''
    Plot classifier likelihoods over different classes for events of a given particle type

    Args:
        softmaxes           ... 2d array with the first dimension n_samples
        labels              ... 1d array of particle labels to use in every output plot, or a list of 4 lists of particle names to use in each respectively
        particle_names      ... list of string names of particle types to plot. All must be keys in 'label_dict' 
        label_dict          ... dictionary of particle labels, with string particle name keys and values corresponding to values taken by 'labels'
        bins                ... optional, number of bins for histogram
        legend_locs         ... list of 4 strings for positioning the legends
        extra_panes         ... list of lists of particle names, each of which contains the names of particles to use in a joint response plot
        xlim                ... limit the x-axis
        label_size          ... font size
        legend_label_dict   ... dictionary of display symbols for each string label, to use for displaying pretty characters
        show                ... if True then display figure, otherwise return figure
    author: Calum Macdonald
    June 2020
    '''
    
    if legend_label_dict is None:
        legend_label_dict = {}
        for name in particle_names:
            legend_label_dict[name] = name

    num_panes = softmaxes.shape[1] + len(extra_panes)

    # Define a set of fresh, crisp colors
    colors = ['tomato','limegreen']
    edgecolors = ['r','g']

    fig, axes = plt.subplots(1, num_panes, figsize=(6 * num_panes, 6), facecolor='white')

    inverse_label_dict = {value: key for key, value in label_dict.items()}

    softmaxes_list = separate_particles([softmaxes], labels, label_dict, [name for name in label_dict.keys()])[0]

    if isinstance(particle_names[0], str):
        particle_names = [particle_names for _ in range(num_panes)]

    fig.suptitle(title, y=0.98, fontsize=label_size + 4)

    # Generate single particle plots
    i = 0
    for independent_particle_label, ax in enumerate(axes[:softmaxes.shape[1]]):
        dependent_particle_labels = [label_dict[particle_name] for particle_name in particle_names[independent_particle_label]]
        if (i>0): 
            dependent_particle_labels = list(reversed(dependent_particle_labels))
            colors = list(reversed(colors))
            edgecolors = list(reversed(edgecolors))
        color_cycle = cycle(colors)  # Cycle through fresh colors for each plot
        edgecolor_cycle=cycle(edgecolors)
        for dependent_particle_label in dependent_particle_labels:
            color=next(color_cycle)
            edgecolor=next(edgecolor_cycle)
            ax.hist(softmaxes_list[dependent_particle_label][:, independent_particle_label],
                    label=f"{legend_label_dict[inverse_label_dict[dependent_particle_label]]} Events",
                    alpha=0.6, histtype='stepfilled',stacked=True, bins=bins, density=True,color=color,edgecolor=edgecolor,
                    linewidth=2)
        ax.legend(loc=legend_locs[independent_particle_label] if legend_locs is not None else 'best', fontsize=label_size)
        ax.set_xlabel('$P_{%s}$'%(legend_label_dict[inverse_label_dict[independent_particle_label]]), fontsize=label_size + 2)
        ax.set_ylabel('Normalised Density', fontsize=label_size + 2)
        ax.set_yscale('log')
        i+=1
    ax = axes[-1]

    # Generate joint plots
    for n, extra_pane_particle_names in enumerate(extra_panes):
        pane_idx = softmaxes.shape[1] + n
        ax = axes[pane_idx]
        dependent_particle_labels = [label_dict[particle_name] for particle_name in particle_names[pane_idx]]
        
        color_cycle = cycle(colors)
        for dependent_particle_label in dependent_particle_labels:
            color=next(color_cycle)
            edgecolor=next(edgecolor_cycle)
            ax.hist(reduce(lambda x, y: x + y, [softmaxes_list[dependent_particle_label][:, label_dict[pname]] for pname in extra_pane_particle_names]),
                    label=legend_label_dict[particle_names[-1][dependent_particle_label]],
                    alpha=0.6, histtype='stepfilled',stacked=True, bins=bins, density=True,color=color,edgecolor=edgecolor,
                    linewidth=2)
        ax.legend(loc=legend_locs[-1] if legend_locs is not None else 'best', fontsize=label_size)
        xlabel = ''
        for list_index, independent_particle_name in enumerate(extra_pane_particle_names):
            xlabel += 'P({})'.format(legend_label_dict[independent_particle_name])
            if list_index < len(extra_pane_particle_names) - 1:
                xlabel += ' + '
        ax.set_xlabel(xlabel, fontsize=label_size + 2)
        ax.set_ylabel('Normalised Density', fontsize=label_size + 2)
        ax.set_yscale('log')

    plt.tight_layout()
    fig.savefig("figures/Classification for Model %s"%(model), dpi=1200, bbox_inches='tight')
    if show:
        plt.show()
        return
    
    return fig

def separate_particles(input_array_list, labels, index_dict, desired_labels=['gamma','e','mu']):
    '''
    Separates all arrays in a list by indices where 'labels' takes a certain value, corresponding to a particle type.
    
    Args:
        input_array_list    ... list of arrays to be separated, must have same length and same length as 'labels'
        labels              ... list of labels, taking any of the three values in index_dict.values()
        index_dict          ... dictionary of particle labels, must have 'gamma','mu','e' keys pointing to values taken by 'labels', 
                                        unless desired_labels is passed
        desired_labels      ... optional list specifying which labels are desired and in what order. Default is ['gamma','e','mu']
    
    Returns: a list of tuples, each tuple contains section of each array corresponsing to a desired label
    author: Calum Macdonald
    June 2020
    '''

    idxs_list = [np.where(labels==index_dict[label])[0] for label in desired_labels]

    separated_arrays = []
    for array in input_array_list:
        separated_arrays.append(tuple([array[idxs] for idxs in idxs_list]))

    return separated_arrays

def compute_roc(softmax_out_val, labels_val, true_label, false_label):
    """
    Compute ROC metrics from softmax and labels for given particle labels

    Args:
        softmax_out_val     ... array of softmax outputs
        labels_val          ... 1D array of actual labels
        true_label          ... label of class to be used as true binary label
        false_label         ... label of class to be used as false binary label
    
    Returns:
        fpr, tpr, thr       ... false positive rate, true positive rate, thresholds used to compute scores
    """
    labels_val_for_comp = labels_val[np.where( (labels_val==false_label) | (labels_val==true_label)  )]
    softmax_out_for_comp = softmax_out_val[np.where(  (labels_val==false_label) | (labels_val==true_label)  )][:,true_label]

    fpr, tpr, thr = roc_curve(labels_val_for_comp, softmax_out_for_comp, pos_label=true_label, drop_intermediate=False)
    
    return fpr, tpr, thr

#def plot_roc(fpr,tpr,thr,fpr2,tpr2,thr2,fpr3,tpr3,thr3,fpr4,tpr4,thr4,true_label_name,false_label_name,fig_list=None,xlims=None,ylims=None,axes=None,linestyle=None,linecolor=None,plot_label=None,show=False):
def plot_roc(fpr_first,tpr_first,thr_first,fpr_angle,tpr_angle,thr_angle,true_label_name,false_label_name,fig_list=None,xlims=None,ylims=None,axes=None,linestyle=None,linecolor=None,plot_label=None,show=False):
    """
    Plot ROC curves for a classifier that has been evaluated on a validation set with respect to given labels
    
    Args:
        fpr, tpr, thr           ... false positive rate, true positive rate, thresholds used to compute scores
        true_label_name         ... name of class to be used as true binary label
        false_label_name        ... name of class to be used as false binary label
        fig_list                ... list of indexes of ROC curves to plot
        xlims                   ... xlims to apply to plots
        ylims                   ... ylims to apply to plots
        axes                    ... axes to plot on
        linestyle, linecolor    ... line style and color
        plot_label              ... string to use in title of plots
        show                    ... if true then display figure, otherwise return figure
    """
    # Compute additional parameters
    # No neutron tag cut
    #eff_dsnb=0.781686
    #eff_ncqe=0.118331
    # With neutron tag cut
    eff_dsnb = 0.798094
    eff_ncqe = 0.105921
    rejection_ncqe = 1.0/eff_ncqe
    tpr_cuts = 0.925828
    fpr_cuts=0.176902


    #tpr_angle*=tpr_cuts
    #fpr_angle*=fpr_cuts

    rejection_first=1.0/fpr_first
    rejection_angle=1.0/fpr_angle
    #rejection3=1.0/(fpr3)
    #rejection4=1.0/(fpr4)
    #rejection5=1.0/(fpr5)
    #rejection6=1.0/(fpr6)
    #linecolor = '#ff7f0e'
    roc_AUC = auc(fpr_first,tpr_first)
    roc_AUC2 = auc(fpr_angle,tpr_angle)
    #roc_AUC3 = auc(fpr3,tpr3)
    #roc_AUC4 = auc(fpr4,tpr4)
    #roc_AUC5 = auc(fpr5,tpr5)
    #roc_AUC6 = auc(fpr6,tpr6)
    signal=tpr_first*0.78
    signal2=tpr_angle*0.78
    linecolor = ["b", "r", 'g', 'y','m','k']
    if fig_list is None:
        fig_list = list(range(4))
    
    figs = []
    # Plot results
    if axes is None:
        if 0 in fig_list:
            fig0, ax0 = plt.subplots(figsize=(12,8),facecolor="w")
            figs.append(fig0)
        if 1 in fig_list: 
            fig1, ax1 = plt.subplots(figsize=(12,8),facecolor="w")
            figs.append(fig1)
        if 2 in fig_list: 
            fig2, ax2 = plt.subplots(figsize=(12,8),facecolor="w")
            figs.append(fig2)
        if 3 in fig_list: 
            fig3, ax3 = plt.subplots(figsize=(12,8),facecolor="w")
            figs.append(fig3)
    else:
        print(axes)
        axes_iter = iter(axes)
        if 0 in fig_list:
            ax0 = next(axes_iter)
        if 1 in fig_list: 
            ax1 = next(axes_iter)
        if 2 in fig_list: 
            ax2 = next(axes_iter)
        if 3 in fig_list: 
            ax3 = next(axes_iter)

    if xlims is not None:
        xlim_iter = iter(xlims)
    if ylims is not None:
        ylim_iter = iter(ylims)

    if 0 in fig_list: 
        ax0.tick_params(axis="both", labelsize=20)
        ax0.plot(fpr_first, tpr_first,
                    label=plot_label if plot_label  is not None else r'FirstRed, AUC={:.3f}'.format(roc_AUC),
                    linestyle=linestyle  if linestyle is not None else None,
                    color=linecolor[0] if linecolor[0] is not None else None,
                    linewidth=2)
        ax0.plot(fpr_angle, tpr_angle,
                    label=plot_label if plot_label  is not None else r'AngleRed, AUC={:.3f}'.format(roc_AUC2),
                    linestyle=linestyle  if linestyle is not None else None,
                    color=linecolor[1] if linecolor[1] is not None else None,
                    linewidth=2)
        #ax0.plot(fpr3, tpr3,
        #            label=plot_label if plot_label  is not None else r'AngleRed, AUC={:.3f}'.format(roc_AUC3),
        #            linestyle=linestyle  if linestyle is not None else None,
        #            color=linecolor[2] if linecolor[2] is not None else None,
        #            linewidth=2)
        #ax0.plot(fpr4, tpr4,
        #            label=plot_label if plot_label  is not None else r'AngleRed_Ndet, AUC={:.3f}'.format(roc_AUC4),
        #            linestyle=linestyle  if linestyle is not None else None,
        #            color=linecolor[3] if linecolor[3] is not None else None)
        #ax0.plot(fpr5, tpr5,
        #            label=plot_label if plot_label  is not None else r'200k, AUC={:.3f}'.format(roc_AUC5),
        #            linestyle=linestyle  if linestyle is not None else None,
        #            color=linecolor[4] if linecolor[4] is not None else None)
        #ax0.plot(fpr6, tpr6,
        #            label=plot_label if plot_label  is not None else r'492k, AUC={:.3f}'.format(roc_AUC5),
        #            linestyle=linestyle  if linestyle is not None else None,
        #            color=linecolor[5] if linecolor[5] is not None else None)
        ax0.plot([eff_ncqe],[eff_dsnb],linestyle='None',marker='x', color = 'black',markeredgewidth=2, markersize = '20',label=r'Traditional Cuts'.format(true_label_name, false_label_name))
        ax0.set_xlabel(f'{false_label_name} Background Efficiency (FPR)', fontsize=20)
        ax0.set_ylabel(f'{true_label_name} Signal Efficiency (TPR)', fontsize=20)
        ax0.legend(loc="lower right",prop={'size': 16})

        if xlims is not None:
            xlim = next(xlim_iter)
            ax0.set_xlim(xlim[0],xlim[1])
        if ylims is not None:
            ylim = next(ylim_iter)
            ax0.set_ylim(ylim[0],ylim[1])

    if 1 in fig_list: 
        ax1.tick_params(axis="both", labelsize=20)
        ax1.set_yscale('log')
        ax1.grid(visible=True, which='major', color='gray', linestyle='-')
        ax1.grid(visible=True, which='minor', color='gray', linestyle='--')
        ax1.set_facecolor((0.95, 0.95, 0.95))
        ax1.plot(tpr_first, rejection_first, 
                    label=plot_label + ', AUC={:.3f}'.format(roc_AUC)  if plot_label is not None else r'FirstRed, AUC={:.3f}'.format(roc_AUC),
                    linestyle=linestyle  if linestyle is not None else None,
                    color=linecolor[0] if linecolor[0] is not None else None,
                    linewidth=2)
        ax1.plot(tpr_angle, rejection_angle, 
                    label=plot_label + ', AUC={:.3f}'.format(roc_AUC2)  if plot_label is not None else r'AngleRed, AUC={:.3f}'.format(roc_AUC2),
                    linestyle=linestyle  if linestyle is not None else None,
                    color=linecolor[1] if linecolor[1] is not None else None,
                    linewidth=2)          
        #ax1.plot(tpr3, rejection3, 
        #            label=plot_label + ', AUC={:.3f}'.format(roc_AUC3)  if plot_label is not None else r'AngleRed, AUC={:.3f}'.format(roc_AUC3),
        #            linestyle=linestyle  if linestyle is not None else None,
        #            color=linecolor[2] if linecolor[2] is not None else None,
        #            linewidth=2)
        #ax1.plot(tpr4, rejection4, 
        #            label=plot_label + ', AUC={:.3f}'.format(roc_AUC4)  if plot_label is not None else r'AngleRed_Ndet, AUC={:.3f}'.format(roc_AUC4),
        #            linestyle=linestyle  if linestyle is not None else None,
        #            color=linecolor[3] if linecolor[3] is not None else None)
        #ax1.plot(tpr5, rejection5, 
        #            label=plot_label + ', AUC={:.3f}'.format(roc_AUC5)  if plot_label is not None else r'200k AUC={:.3f}'.format(roc_AUC5),
        #            linestyle=linestyle  if linestyle is not None else None,
        #            color=linecolor[4] if linecolor[4] is not None else None)
        #ax1.plot(tpr6, rejection6, 
        #            label=plot_label + ', AUC={:.3f}'.format(roc_AUC5)  if plot_label is not None else r'492k AUC={:.3f}'.format(roc_AUC5),
        #            linestyle=linestyle  if linestyle is not None else None,
        #            color=linecolor[5] if linecolor[5] is not None else None)
        #'''ax1.plot(tpr3, rejection3, 
        #            label=plot_label + ', AUC={:.3f}'.format(roc_AUC3)  if plot_label is not None else r'BDT no nhit, AUC={:.3f}'.format(roc_AUC3),
        #            linestyle=linestyle  if linestyle is not None else None,
        #            color=linecolor if linecolor is not None else None)'''
        #ax1.plot([0.132535],[rejectionL],linestyle='None',marker='x', color = 'red', markersize = '20',label=r'Solar Analysis Cuts'.format(true_label_name, false_label_name))
        ax1.plot([eff_dsnb],[rejection_ncqe],linestyle='None',marker='x', color = 'black', markeredgewidth=2,markersize = '20',label=r'Traditional Cuts'.format(true_label_name, false_label_name))
        xlabel = f'{true_label_name} Signal Efficiency (TPR)'
        ylabel = f'{false_label_name} Background Rejection Rate (1/FPR)'
        title = '{} vs {} Rejection (Testing Set: AngleRed)'.format(true_label_name, false_label_name)
        #ax1.set_ylim(0,1)
        ax1.set_xlabel(xlabel, fontsize=20)
        ax1.set_ylabel(ylabel, fontsize=20)
        ax1.set_title(title, fontsize=24)
        ax1.legend(loc="upper right",prop={'size': 16}) #bbox_to_anchor=(1.05, 1), loc='upper left') #loc="upper right",prop={'size': 16})

        if xlims is not None:
            xlim = next(xlim_iter)
            ax1.set_xlim(xlim[0],xlim[1])
        if ylims is not None:
            ylim = next(ylim_iter)
            ax1.set_ylim(ylim[0],ylim[1])
    
    if 2 in fig_list: 
        ax2.tick_params(axis="both", labelsize=20)
        plt.yscale('log')
        #plt.ylim(1.0,1)
        ax2.grid(visible=True, which='major', color='gray', linestyle='-')
        ax2.grid(visible=True, which='minor', color='gray', linestyle='--')
        ax2.plot(tpr_first, tpr_first/np.sqrt(fpr_first), 
                    label= plot_label + ', AUC={:.3f}'.format(roc_AUC) if plot_label is not None else r'{} VS {} FirstRed, AUC={:.3f}'.format(true_label_name,false_label_name,roc_AUC),
                    linestyle=linestyle  if linestyle is not None else None,
                    color=linecolor[0] if linecolor[0] is not None else None,
                    linewidth=2)
        ax2.plot(tpr_angle, tpr_angle/np.sqrt(fpr_angle), 
                    label= plot_label + ', AUC={:.3f}'.format(roc_AUC2) if plot_label is not None else r'{} VS {} AngleRed, AUC={:.3f}'.format(true_label_name,false_label_name,roc_AUC2),
                    linestyle=linestyle  if linestyle is not None else None,
                    color=linecolor[1] if linecolor[1] is not None else None,
                    linewidth=2)
        #ax2.plot(tpr3, tpr3/np.sqrt(fpr3), 
        #            label= plot_label + ', AUC={:.3f}'.format(roc_AUC3) if plot_label is not None else r'{} VS {} AngleRed, AUC={:.3f}'.format(true_label_name,false_label_name,roc_AUC3),
        #            linestyle=linestyle  if linestyle is not None else None,
        #            color=linecolor[2] if linecolor[2] is not None else None,
        #            linewidth=2)
        #ax2.plot(tpr4, tpr4/np.sqrt(fpr4), 
        #            label= plot_label + ', AUC={:.3f}'.format(roc_AUC4) if plot_label is not None else r'{} VS {} AngleRed_Ndet, AUC={:.3f}'.format(true_label_name,false_label_name,roc_AUC4),
        #            linestyle=linestyle  if linestyle is not None else None,
        #            color=linecolor[3] if linecolor[3] is not None else None,
        #            linewidth=2)
        #ax2.plot(tpr5, tpr5/np.sqrt(fpr5), 
        #            label= plot_label + ', AUC={:.3f}'.format(roc_AUC4) if plot_label is not None else r'{} VS {} 200k, AUC={:.3f}'.format(true_label_name,false_label_name,roc_AUC4),
        #            linestyle=linestyle  if linestyle is not None else None,
        #            color=linecolor[4] if linecolor[4] is not None else None,
        #            linewidth=2)
        #ax2.plot(tpr6, tpr6/np.sqrt(fpr6), 
        #            label= plot_label + ', AUC={:.3f}'.format(roc_AUC4) if plot_label is not None else r'{} VS {} 492k, AUC={:.3f}'.format(true_label_name,false_label_name,roc_AUC4),
        #            linestyle=linestyle  if linestyle is not None else None,
        #            color=linecolor[5] if linecolor[5] is not None else None,
        #            linewidth=2)
        ax2.set_ylabel('~significance', fontsize=20)
        ax2.legend(loc="upper right",prop={'size': 16})

        if xlims is not None:
            xlim = next(xlim_iter)
            ax2.set_xlim(xlim[0],xlim[1])
        if ylims is not None:
            ylim = next(ylim_iter)
            ax2.set_ylim(ylim[0],ylim[1])
            
        
    if 3 in fig_list: 
        plt.yscale('log')
        ax3.tick_params(axis="both", labelsize=20)
        ax3.grid(visible=True, which='major', color='gray', linestyle='-')
        ax3.grid(visible=True, which='minor', color='gray', linestyle='--')
        ax3.plot(thr_first, fpr_first, 
                    label=plot_label + ', AUC={:.3f}'.format(roc_AUC)  if plot_label is not None else r'FirstRed, AUC={:.3f}'.format(roc_AUC),
                    linestyle=linestyle  if linestyle is not None else None,
                    color=linecolor[0] if linecolor[0] is not None else None,
                    linewidth=2)
        ax3.plot(thr_angle, fpr_angle, 
                    label=plot_label + ', AUC={:.3f}'.format(roc_AUC2)  if plot_label is not None else r'AngleRed, AUC={:.3f}'.format(roc_AUC2),
                    linestyle=linestyle  if linestyle is not None else None,
                    color=linecolor[1] if linecolor[1] is not None else None,
                    linewidth=2)          
        #ax3.plot(thr3, fpr3, 
        #            label=plot_label + ', AUC={:.3f}'.format(roc_AUC3)  if plot_label is not None else r'Hybrid - dummy eval, AUC={:.3f}'.format(roc_AUC3),
        #            linestyle=linestyle  if linestyle is not None else None,
        #            color=linecolor if linecolor is not None else None)
        #ax3.plot(thr4, fpr4, 
        #            label=plot_label + ', AUC={:.3f}'.format(roc_AUC4)  if plot_label is not None else r'Hybrid - mrg eval, AUC={:.3f}'.format(roc_AUC4),
        #            linestyle=linestyle  if linestyle is not None else None,
        #            color=linecolor if linecolor is not None else None)
        #ax3.plot(thr5, fpr5, 
        #            label=plot_label + ', AUC={:.3f}'.format(roc_AUC4)  if plot_label is not None else r'Hybrid - mrg eval, AUC={:.3f}'.format(roc_AUC4),
        #            linestyle=linestyle  if linestyle is not None else None,
        #            color=linecolor if linecolor is not None else None)

        '''    ax1.plot(tpr3, rejection3, 
                    label=plot_label + ', AUC={:.3f}'.format(roc_AUC3)  if plot_label is not None else r'BDT no nhit, AUC={:.3f}'.format(roc_AUC3),
                    linestyle=linestyle  if linestyle is not None else None,
                    color=linecolor if linecolor is not None else None)'''
        xlabel = f'Network Cut Value'
        ylabel = f'{true_label_name} Background Efficiency'
        ax3.set_xlim(0.9, 1.0)
        ax3.set_xlabel(xlabel, fontsize=20)
        ax3.set_ylabel(ylabel, fontsize=20)
        ax3.legend(loc="upper right",prop={'size': 16}) #bbox_to_anchor=(1.05, 1), loc='upper left') #loc="upper right",prop={'size': 16})

        #ax3.set_xlim(.998,1)
        #ax3.set_ylim(0,.)
        


    if show:
        plt.show()
        return
    
    if axes is None:
        return tuple(figs)

def plot_rocs(first_softmax_out_val,first_labels_val,angle_softmax_out_val,angle_labels_val,labels_dict,plot_list=None,vs_list = None,show=True,overlay=False):
#def plot_rocs(softmax_out_val,labels_val,softmax_out_val2,labels_val2,softmax_out_val3,labels_val3,softmax_out_val4,labels_val4,labels_dict,plot_list=None,vs_list = None,show=True,overlay=False):
    """
    Plot ROC curves for a classifier for a series of combinations of labels
    
    Args:
        softmax_out_val     ... array of softmax outputs
        labels_val          ... actual labels
        labels_dict         ... dict matching particle labels to numerical labels
        plot_list           ... list of labels to use as true labels
        vs_list             ... list of labels to use as false labels
        show                ... if true then display figure, otherwise return figure
    """
    # if no list of labels to plot, assume using all members of dict
    all_labels = list(labels_dict.keys())

    if plot_list is None:
        plot_list = all_labels
    
    if vs_list is None:
        vs_list = all_labels
    
    figs = []
    plot_name = ['ROC', 'ROC_inverse', 'significance']
    # plot ROC curves for each specified label
    for true_label_name in plot_list:
        true_label = labels_dict[true_label_name]
        for false_label_name in vs_list:
            false_label = labels_dict[false_label_name]
            if not (false_label_name == true_label_name):
                # initialize figure
                #num_panes = 3
                #fig, axes = plt.subplots(1, num_panes, figsize=(8*num_panes,12), facecolor='w')
                #fig.suptitle("ROC for {} vs {}".format(true_label_name, false_label_name), fontweight='bold',fontsize=32)
                fpr_first,tpr_first,thr_first=compute_roc(first_softmax_out_val,first_labels_val, true_label, false_label)
                fpr_angle,tpr_angle,thr_angle=compute_roc(angle_softmax_out_val,angle_labels_val, true_label, false_label)
                #fpr3,tpr3,thr3=compute_roc(softmax_out_val3, labels_val3, true_label, false_label)
                #fpr4,tpr4,thr4=compute_roc(softmax_out_val4, labels_val4, true_label, false_label)
                #fpr5,tpr5,thr5=compute_roc(softmax_out_val5, labels_val5, true_label, false_label)
                #fpr6,tpr6,thr6=compute_roc(softmax_out_val6, labels_val6, true_label, false_label)
                #roc_figs=plot_roc(fpr,tpr,thr,fpr2,tpr2,thr2,fpr3,tpr3,thr3,fpr4,tpr4,thr4,fpr5,tpr5,thr5,true_label_name,false_label_name, fig_list=[0,1,3]#xlims=[[0,1],[0,1],[0,1]])
                #roc_figs=plot_roc(fpr,tpr,thr,fpr2,tpr2,thr2,fpr3,tpr3,thr3,true_label_name,false_label_name, fig_list=[0,1])#xlims=[[0,1],[0,1],[0,1]])
                roc_figs=plot_roc(fpr_first,tpr_first,thr_first,fpr_angle,tpr_angle,thr_angle,true_label_name,false_label_name,fig_list=[0,1,2])#xlims=[[0,1],[0,1],[0,1]])
                #plot_roc(softmax_out_val, labels_val, true_label_name, true_label, false_label_name, false_label, axes=axes)
                for idx,fig in enumerate(roc_figs):
                    fig_name = f"/opt/ppd/hyperk/Users/samanis/WatChMaL/analysis/figures/{true_label_name}_vs_{false_label_name}_{plot_name[idx]}.png"
                    fig.savefig(fig_name)
                    figs.append(fig)
                
                #figs.append(fig)
    if show:
        plt.show()
        return
    
    #return figs
path='/opt/ppd/hyperk/Users/samanis/WatChMaL/outputs/resnet18/'
fir_softmaxes_fir=np.load("%s/firstred_ndet/anglered_test/softmax.npy"%(path))
fir_labels_fir=np.load("%s/firstred_ndet/anglered_test/labels.npy"%(path))
fir_predictions_fir=np.load("%s/firstred_ndet/anglered_test/predictions.npy"%(path))

ang_softmaxes_fir=np.load("%s/anglered_ndet/softmax.npy"%(path))
ang_labels_fir=np.load("%s/anglered_ndet/labels.npy"%(path))
ang_predictions_fir=np.load("%s/anglered_ndet/predictions.npy"%(path))

#softmaxes5=np.load("%s/200k/softmax.npy"%(path))
#labels5=np.load("%s/200k/labels.npy"%(path))
#softmaxes6=np.load("%s/492k/softmax.npy"%(path))
#labels6=np.load("%s/492k/labels.npy"%(path))




#plot_classifier_response(softmaxes,labels,['Data','8B'],{'8B':1,'Data':0},legend_locs = ['upper center','upper center'], bins = 50)
#plot_classifier_response(softmaxes2,labels2,['Data','8B'],{'8B':1,'Data':0},legend_locs = ['upper center','upper center'], bins = 50)
#plot_classifier_response(softmaxes3,labels3,['Data','8B'],{'8B':1,'Data':0},legend_locs = ['upper center','upper center'], bins = 50)
#plot_classifier_response(softmaxes4,labels4,['Data','8B'],{'8B':1,'Data':0},legend_locs = ['upper center','upper center'], bins = 50)

#plot_classifier_response(fir_softmaxes_fir,fir_labels_fir,['NCQE','DSNB'],{'DSNB':0,'NCQE':1},legend_locs = ['upper center','upper center'], bins = 50, title = "FirstRed CNN tested on FirstRed Test Set", model = 'FirstRed')
#plot_classifier_response(ang_softmaxes_fir,ang_labels_fir,['NCQE','DSNB'],{'DSNB':0,'NCQE':1},legend_locs = ['upper center','upper center'], bins = 50, title = "AngleRed CNN tested on AngleRed Test Set", model = 'AngleRed')
#plot_confusion_matrix(fir_predictions_fir,fir_labels_fir, ['DSNB', 'NCQE'],title = "FirstRed (Model) tested on FirstRed (Sample)")
#plot_confusion_matrix(ang_predictions_fir,ang_labels_fir, ['DSNB', 'NCQE'],title = "AngleRed (Model) tested on FirstRed (Sample)")
##plot_rocs(softmaxes,labels,softmaxes2,labels2,softmaxes3,labels3,softmaxes4,labels4,softmaxes5,labels5,{'8B':1,'Data':0},plot_list=['8B'],vs_list=['Data'],show=True)
plot_rocs(fir_softmaxes_fir,fir_labels_fir,ang_softmaxes_fir,ang_labels_fir,{'DSNB':1,'NCQE':0},plot_list=['DSNB'],vs_list=['NCQE'],show=True)
