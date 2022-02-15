from __future__ import print_function
import numpy as np
import itertools
import os
import matplotlib.pyplot as plt
from pathlib import Path
from IPython import display
from IPython.display import Video
import sys
sys.path.append(f'../src')
import sac_tri

#a constant
PLOT_DIR_NAME = "plots"

"""
This module contains functions to visualize data that was logged with sac_multi.SacTrain and by
sac_tri.SacTrain. In includes visualizing plots in Jupyter notebook, exporting them as pdfs.
"""

#most useful function

def plot_sac_logs(log_dir, running_reward_file=None, running_loss_file=None, actions_file=None,
                running_multi_obj_file=None,actions_to_plot=400,plot_to_file_line = None, suppress_show=False,
                save_plot = False, actions_per_log=2000, extra_str="", is_tri=False, actions_ylim=None,
                dont_clear_output=False):
    """
    Produces and displays in a Jupyter notebook a single plot with the running reward, running multi objectives,
    the loss function, the entropy, the running value of the weight c (if present) and the last chosen actions.
    This function can also save the plot to .pdf in the PLOT_DIR_NAME folder inside the log folder.

    Args:
        log_dir (str): location of the folder with all the logging
        running_reward_file (str): location of the txt file with the running reward. If None, default location is used
        running_loss_file (str): location of the txt file with the loss function. If None, default location is used
        actions_file (str): location of the txt file with the actions. If None, default location is used
        running_multi_obj_file (str): location of the txt file with the running multi objectives. If Nonde, default
            location is used
        actions_to_plot (int): how many of the last actions to show
        plot_to_file_line (int): plots logs only up to a given file line. In None, all data is used
        suppress_show (bool): if True, it won't display the plot
        save_plot (bool): If True, is saved the plot as a pdf in PLOT_DIR_NAME in the log folder
        actions_per_log (int): number of actions taken between logs. Corresponds to LOG_STEPS hyperparam while training
        extra_str (str): string to append to the file name of the saved plot
        is_tri (bool): if the logged data has the discrete action (True) or not (False)
        actions_ylim (tuple): a 2 element tuple specifying the y_lim of the actions plot
        dont_clear_output (bool): if False, it will clear the previous plots shown with this function
    """
    #create the file locations if they are not passed it
    running_reward_file, running_loss_file, running_multi_obj_file, actions_file = \
                                log_sac_file_locations(log_dir, running_reward_file, running_loss_file,
                                                        running_multi_obj_file, actions_file)
    #check if the files exist
    running_reward_exists = Path(running_reward_file).exists()
    running_loss_exists = Path(running_loss_file).exists()
    actions_exists = Path(actions_file).exists()
    running_multi_obj_quantities = count_quantities(running_multi_obj_file)
    loss_elements = count_quantities(running_loss_file)
    
    #count the number of plots to do
    quantities_to_log = int(running_reward_exists) + loss_elements*int(running_loss_exists) + int(actions_exists) + \
        running_multi_obj_quantities
   
    #create the matplotlib subplots
    fig, axes = plt.subplots(quantities_to_log, figsize=(7,quantities_to_log*2.2))
    if quantities_to_log == 1:
        axes = [axes]
    axis_ind = 0
    
    #plot the running reward
    if running_reward_exists:
        plot_running_reward_on_axis(running_reward_file, axes[axis_ind], plot_to_file_line)
        axis_ind += 1

    #plot the running multi objectives
    if running_multi_obj_quantities > 0:
            plot_running_multi_obj_on_axes(running_multi_obj_file, axes[axis_ind : axis_ind+running_multi_obj_quantities],
                                                plot_to_file_line)
            axis_ind += running_multi_obj_quantities

    #plot the running loss
    if running_loss_exists:
        plot_running_loss_on_axis(running_loss_file, axes[axis_ind:axis_ind+loss_elements], plot_to_file_line,is_tri=is_tri)
        axis_ind += loss_elements

    #plot the last actions
    if actions_exists:
            plot_actions_on_axis(actions_file, axes[axis_ind],actions_to_plot=actions_to_plot,
                    plot_to_file_line= None if plot_to_file_line is None else int(plot_to_file_line*actions_per_log),
                    is_tri=is_tri, actions_ylim=actions_ylim )
            axis_ind += 1 
    
    #compact view
    fig.tight_layout()  

    #save the plot if requested
    if save_plot:
        plot_folder = log_dir + PLOT_DIR_NAME
        Path(plot_folder).mkdir(parents=True, exist_ok=True)
        plot_file_name = os.path.join(plot_folder, f"plt{extra_str}.pdf")
        fig.savefig(plot_file_name, bbox_inches='tight')

    #display the plot if requested
    if not suppress_show:
        if not dont_clear_output:
            display.clear_output(wait=True)
        display.display(plt.gcf())
    plt.close() 

#functions mainly used internally

def log_sac_file_locations(log_dir, running_reward_file, running_loss_file,running_multi_obj_file, actions_file):
    """
    Returns the location of the logging files. If they are passed, it doesn't change them.
    If they are None, it returns the default location

    Args:
        log_dir (str): location of the logging folder
        running_reward_file (str): location of reward file. If None, default location is returned
        running_loss_file (str): location of loss file. If None, default location is returned
        running_multi_obj_file (str): location of multi_obj file. If None, default location is returned
        actions_file (str): location of actions file. If None, default location is returned

    Returns:
        running_reward_file (str): location of reward file
        running_loss_file (str): location of loss file
        running_multi_obj_file (str): location of the multi_obj file
        actions_file (str): location of actions file
    """
    if running_reward_file is None:
        running_reward_file = os.path.join(log_dir, sac_tri.SacTrain.RUNNING_REWARD_FILE_NAME)
    if running_loss_file is None:
        running_loss_file = os.path.join(log_dir, sac_tri.SacTrain.RUNNING_LOSS_FILE_NAME)
    if running_multi_obj_file is None:
        running_multi_obj_file = os.path.join(log_dir, sac_tri.SacTrain.RUNNING_MULTI_OBJ_FILE_NAME)
    if actions_file is None:
        actions_file = os.path.join(log_dir, sac_tri.SacTrain.ACTIONS_FILE_NAME)
    return (running_reward_file, running_loss_file, running_multi_obj_file, actions_file)

def plot_running_reward_on_axis(file_location, axis, plot_to_file_line = None,multi_location=None,ylabel = None,
                                 xlabel = None, xticks = None, xticklabels=None, yticks = None, yticklabels=None,
                                 k_notation=True, linewidth=None, custom_colors=None, lines_to_mark = None, custom_mark_color = None,
                                 custom_mark_size=None,ylim=None,plot_extra_args=None,plot_extra_kwargs=None,legend_labels=None,
                                 legend_cols=1,legend_location="best",extra_coeffs=None,legend_column_spacing=None,
                                 legend_fancybox=False,legend_framealpha=0., skip_lines=None):
    """
    Produces a plot of the running reward on a given matplot lib axis

    Args:
        file_location (str): location of the file with the running reward
        axis (matplotlib axis): the axis on which to do the plot
        plot_to_file_line (int): plot data up to this file line. If None, plots till the end
        multi_location (str): optional. If specificed, it also plots the multiobjectives
        ylabel (str): custom string for y axis
        xlabel (str): custom string for x axis
        xticks (list(float)): custom list of x ticks
        xticklabels (list(str)): custom list of x tick strings
        yticks (list(float)): custom list of y ticks
        yticklabels (list(str)): custom list of y tick strings
        k_notation (bool): if True, displays number of x axis using "k" to represent 1000
        linewidth (float): linewidth of line
        custom_colors (tuple(str)): list of colors to use for the plot lines
        lines_to_mark (list(int)): adds a circle around the points corresponding to the specified lines in the file
        custom_mark_color (color): color of the circles at the points specified by lines_to_mark
        custom_mark_size (float): size of the marks
        ylim (tuple(int)): ylim delimiter
        plot_extra_args: will call the function axis.plot passing in these custom args
        plot_extra_kwargs: will call the function axis.plot passing in plot_extra_args and plot_extra_kwargs
        legend_labels (list(str)): list of strings for the legend labels
        legend_cols (int): numer of columns of the legend
        legend_location (str): legend location
        extra_coeffs(tuple(float)): list of floats, each one mutiplying the y coordinate of each plot
        legend_column_spacing: sets the columnspacing for the legend in matplotlib
        legend_fancybox: sets the fancybox for the legend in matplotlib
        legend_framealpha: sets framealpha for the legend in matplotlib
        skip_lines (int): if specified, it skips the first skip_lines lines when loading the data to plot
    """
    #setup the plot
    if legend_labels is None:
        label_iter = itertools.cycle([None])
    else:
        label_iter = itertools.cycle(legend_labels)
    if ylabel is None:
        ylabel = "$G$"
    if xlabel is None:
        xlabel = "step"
    #load the data
    reward_data = np.loadtxt(file_location).reshape(-1,2)
    if skip_lines is not None:
        reward_data = reward_data[skip_lines:]
    if multi_location is not None:
        multi_data = np.loadtxt(multi_location)
        if skip_lines is not None:
            multi_data = multi_data[skip_lines:]
   
    #sets up extra_coeffs
    if extra_coeffs is None:
        extra_coeffs = itertools.cycle([1.])
    else:
        extra_coeffs = itertools.cycle(extra_coeffs)

    #sets up the custom colors
    if custom_colors is None:
        custom_colors = itertools.cycle(["black"])
    else:
        custom_colors = itertools.cycle(custom_colors)

    #sets up the lines to mark
    if lines_to_mark is not None:
        points_to_mark = reward_data[lines_to_mark]
    
    #plot to file line cropping
    if plot_to_file_line is not None:
        reward_data = reward_data[:plot_to_file_line+1]
        if multi_location is not None:
            multi_data = multi_data[:plot_to_file_line+1]
    
    #plot the reward
    axis.plot(reward_data[:,0],reward_data[:,1]*next(extra_coeffs), linewidth=linewidth, color=next(custom_colors), label=next(label_iter))
    
    #plot the various multiojbectives
    if multi_location is not None:
        for i in range(1,multi_data.shape[1]):
            axis.plot(multi_data[:,0],multi_data[:,i]*next(extra_coeffs), linewidth=linewidth, color=next(custom_colors), label=next(label_iter))
    
    #labels
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    if k_notation:
        axis.xaxis.set_major_formatter(lambda x,y: num_to_k_notation(x) )
    
    #marks
    if lines_to_mark is not None:
        axis.scatter(points_to_mark[:,0],points_to_mark[:,1], color=custom_mark_color,s=custom_mark_size,zorder=100)
    
    #ylim
    if ylim is not None:
        axis.set_ylim(ylim)
    
    #extra plots
    if plot_extra_args is not None:
        if plot_extra_kwargs is None:
            axis.plot(*plot_extra_args, label=next(label_iter))
        else:
            axis.plot(*plot_extra_args, **plot_extra_kwargs, label=next(label_iter))
    if xticks is not None:
        axis.set_xticks(xticks)
    if xticklabels is not None:
        axis.set_xticklabels(xticklabels)
    if yticks is not None:
        axis.set_yticks(yticks)
    if yticklabels is not None:
        axis.set_yticklabels(yticklabels)
    if legend_labels is not None:
        axis.legend(loc=legend_location,fancybox=legend_fancybox, framealpha=legend_framealpha,borderaxespad=0.,handlelength=1.1,
         ncol=legend_cols, columnspacing=legend_column_spacing) 

def plot_running_multi_obj_on_axes(file_location, axes, plot_to_file_line = None):
    """
    Produces a plot of the running multiple objective, putting each quantity on a different axis.
    axes must contain the correct number of axis.

    Args:
        file_location (str): location of the file with the running multi objectives
        axes (tuple(matplotlib axis)): list of axis for each objective
        plot_to_file_line (int): plot data up to this file line. If None, plots till the end
    """
    #load the data
    plot_data = np.loadtxt(file_location)
    if len(plot_data.shape) == 1:
        plot_data = plot_data.reshape(1,-1)
    if plot_to_file_line is not None:
        plot_data = plot_data[:plot_to_file_line+1]

    #loop over each quantity to plot
    for i,axis in enumerate(axes):
        #perform the plot and set the labels
        axis.plot(plot_data[:,0],plot_data[:,i+1])
        axis.set_xlabel("step")
        axis.set_ylabel(f"Obj {i}")

def plot_running_loss_on_axis(file_location, axes, plot_to_file_line = None,is_tri=True, y_labels = None,
                                x_labels = None, x_k_notations=None, y_k_notations=None, y_ticks = None,
                                custom_colors=None, y_lims = None, skip_lines = None):
    """
    Produces a plot of the quantities in the running losses files on the provided axis
    Args:
        file_location (str): location of the file with the running losses
        axes (list(matplotlib axis)): list of axes on which to plot the various quantities
        plot_to_file_line (int): plot data up to this file line. If None, plots till the end
        is_tri (bool): if it has the 3 discrete actions or not (the quantites change in this case)
        y_labels (list(str)): if specified, one label for each quantity
        x_labels (list(str)): if specified, one label for each quantity
        k_notations (list(bool)): if specificed, one book for each quantity. When True, displays
            number of x axis using "k" to represent 1000
        y_k_notations (list(bool)): if specificed, one book for each quantity. When True, displays
            number of y axis using "k" to represent 1000
        y_ticks(list(list(float))): if specified, the first index represents a list of y ticks for each quantity
        custom_colors (list(colors)): if specified, it represents one plot line color per quantity
        y_lims (list(tuple(float))): if specified, the first index represents the y lims for that quantity
        skip_lines (list(int)): if specified, it represents the number of lines to skip for each quantity 
            counting from the beginning
    """
    #load data
    plot_data = np.loadtxt(file_location)
    if len(plot_data.shape) == 1:
        plot_data = plot_data.reshape(1,-1)
    if skip_lines is not None:
        plot_data = plot_data[skip_lines:]
    if plot_to_file_line is not None:
        plot_data = plot_data[:plot_to_file_line+1]
    #prepare y labels
    if y_labels is None:
        if is_tri:
            temp_labels = ["alpha D","alpha C", "entropy D", "entropy C" ]
        else:
            temp_labels = ["alpha","entropy", "c weight"]
        y_labels = ["Q Running Loss", "Pi Running Loss"] + temp_labels
        if len(axes) > 6:
            for i in range(6, len(axes)):
                y_labels += [f"loss {i}"]
    #prepare x labels
    if x_labels is None:
        x_labels = ["steps" for _ in range(len(axes)) ]
    #prepare x_k_notations
    if x_k_notations is None:
        x_k_notations = [True for _ in range(len(axes)) ]
    #prepare y_k_notations
    if y_k_notations is None:
        y_k_notations = [False for _ in range(len(axes)) ]
    #prepare y_ticks
    if y_ticks is None:
        y_ticks = [None for _ in range(len(axes))]
    #prepare yscales
    y_scales = ["log", None, "log"]
    y_scales += ["log"] if is_tri else [None]
    y_scales += [None,None]
    if len(axes) > 4:
        y_scales += [[None] for _ in range(4, len(axes))]
    #prepare custom colors
    if custom_colors is None:
        custom_colors = [None for _ in range(len(axes))]
    #prepare y_lims
    if y_lims is None:
        y_lims = [None for _ in range(len(axes))]
  
    #Do all the plots
    for i in range(0,len(axes)):
        #plot data
        axes[i].plot(plot_data[:,0],plot_data[:,i+1], color=custom_colors[i])
        #labels
        axes[i].set_ylabel(y_labels[i])
        axes[i].set_xlabel(x_labels[i])
        #k notations
        if x_k_notations[i]:
            axes[i].xaxis.set_major_formatter(lambda x,y: num_to_k_notation(x) )
        if y_k_notations[i]:
            axes[i].yaxis.set_major_formatter(lambda x,y: num_to_k_notation(x) )
        #y scale
        if y_scales[i] is not None:
            axes[i].set_yscale(y_scales[i])
        #y lims
        if y_lims[i] is not None:
            axes[i].set_ylim(y_lims[i])
        #y ticks
        if y_ticks[i] is not None:
            axes[i].set_yticks(y_ticks[i])

def plot_actions_on_axis(file_location, axis, actions_to_plot=1200, plot_to_file_line = None,
                        is_tri=False,actions_ylim = None, ylabel = None, xlabel = None, xticks = None, xticklabels=None,
                        yticks = None, yticklabels=None,k_notation = True, constant_steps=False, x_count_from_zero=False,
                        custom_colors = None, linewidth=None, two_xticks=False,extra_cycles=None,extra_cycles_linewidth=None,
                        legend_lines=None, legend_text=None, legend_location="best", legend_cols = None, hide_gray_vertical_line=False,
                        legend_line_length=0.5, hide_xaxis_label=False, hide_yaxis_label=False,line_style =None,legend_column_spacing=None ):
    """
    Produces a plot of the last chosen actions on a given matplot lib axis

    Args:
        file_location (str): location of the file with the actions
        axis (matplotlib axis): the axis on which to do the plot
        actions_to_plot (int): how many actions to display in the plot
        plot_to_file_line (int): plot data up to this file line. If None, plots till the end
        is_tri (bool): True if the discrete action is present
        actions_ylim (list(float)): delimiter for the y axis
        ylabel (str): custom string for y axis
        xlabel (str): custom string for x axis
        xticks (list(float)): custom list of x ticks
        xticklabels (list(str)): custom list of x tick strings
        yticks (list(float)): custom list of y ticks
        yticklabels (list(str)): custom list of y tick strings
        k_notation (bool): if True, displays number of x axis using "k" to represent 1000
        constant_steps (bool): if True, it plots the actions as piecewise constant with dashed line
            for the jumps. Otherwise each action is just a dot
        x_count_from_zero (bool): if true, sets x axis from zero. Otherwise from step index
        custom_colors(list(colors)): list of colors for the discrete actions
        linewidth (float): linewidth of line
        two_xticks (bool): if True, it only places 2 x_ticks at smallest and largest value
        extra_cycles: used to overlap additional given functions on the plot. It must be a list of
            tuples as the form (function, range, color).
        extra_cycles_linewidth (float): dith of the extra cycle line
        legend_lines(list(Line2D)): list of Line2D objects to generate the legend
        legend_text (list(str)): list of strings for the legend
        legend_location: matplotlib legend_location
        legend_cols (int): number of columns for the legend
        hide_gray_vertical_line (bool): if True, it doesn't draw a verticle gray line connecting actions
        legend_line_length (float): the lenght of the line in the legend
        hide_xaxis_label (bool): if True, it hides the label for the x axis
        hide_yaxis_label (bool): if True, it hides the label for the y axis
        line_style (str): can be None, or one of the 4 strings: 
            "constant": the action is represented as an horizontal line of length dt
            "scatter": each action is represented as a single dot
            "plot": it plots the actions joining them in the standard way
            "scatter_plot": puts a dot on each action, but also adds vertical gray lines connecting actions
        legend_column_spacing: sets the columnspacing of the legend
    """

    #prepare labels
    if ylabel is None:
        ylabel = "$u$"
    if xlabel is None:
        xlabel = "$t$" if x_count_from_zero else "step"   
    
    #set up the line_style
    if line_style is None:
            line_style = "constant" if constant_steps else "scatter"

    #set the color iterator
    color_iter = None if custom_colors is None else itertools.cycle(custom_colors)

    #prevents weird +e5 from axis labels
    axis.ticklabel_format(useOffset=False)

    #load data
    plot_data = np.loadtxt(file_location)
    if plot_to_file_line is None:
        plot_to_file_line = plot_data.shape[0]-1
    plot_to_file_line = min(plot_to_file_line,plot_data.shape[0]-1)
    actions_to_plot = min(actions_to_plot, plot_data.shape[0])
    data_to_plot = plot_data[(plot_to_file_line-actions_to_plot+1):(plot_to_file_line+1)]
    #if counting from zero, i replace the x values with growing numbers from zero
    if x_count_from_zero:
        data_to_plot[:,0] = np.arange(data_to_plot.shape[0])
    #set ylim
    if actions_ylim is not None:
        axis.set_ylim(actions_ylim)
    #main plotting if there is a discrete action
    if is_tri:
        if line_style == "constant":
            #load colors
            tri_colors = [next_color(color_iter),next_color(color_iter),next_color(color_iter)]
            #compute the step to draw the length of the last point
            dt = data_to_plot[-1,0] - data_to_plot[-2,0]
            #first I do the vertical dahsed line, so they get covered
            if not hide_gray_vertical_line:
                for i in range(data_to_plot.shape[0]-1):
                    axis.plot([data_to_plot[i+1,0],data_to_plot[i+1,0]],[data_to_plot[i,2],data_to_plot[i+1,2]], color="lightgray",
                            linewidth=0.8)
            #horizontal lines
            data_to_plot = np.concatenate( (data_to_plot, np.array([[data_to_plot[-1,0]+dt,0.,0.]]) ))
            for i in range(data_to_plot.shape[0]-1):   
                    axis.plot([data_to_plot[i,0],data_to_plot[i+1,0]],[data_to_plot[i,2],data_to_plot[i,2]], 
                                color=tri_colors[nearest_int(data_to_plot[i,1])], linewidth=linewidth)
        elif line_style == "scatter":
            tri_act_data = data_to_plot[:,1]
            zero_mask = np.abs(tri_act_data) < 0.00001
            one_mask = np.abs(tri_act_data-1.) < 0.00001
            two_mask = np.abs(tri_act_data-2.) < 0.00001
            zero_data = data_to_plot[zero_mask]
            one_data = data_to_plot[one_mask]
            two_data = data_to_plot[two_mask]
            for i in range(2,data_to_plot.shape[1]):
                axis.scatter(zero_data[:,0],zero_data[:,i], color = next_color(color_iter),s=linewidth)
                axis.scatter(one_data[:,0],one_data[:,i], color = next_color(color_iter),s=linewidth)
                axis.scatter(two_data[:,0],two_data[:,i], color = next_color(color_iter),s=linewidth)
        elif line_style == "plot":
            #assume only one continuous action
            last_daction = int(data_to_plot[0,1])
            last_i = 0
            three_colors = [next_color(color_iter) for _ in range(3)]
            for i in range(data_to_plot.shape[0]):
                new_daction = int(data_to_plot[i,1])
                if new_daction != last_daction or i ==data_to_plot.shape[0]-1:
                    x_vals = data_to_plot[last_i:i,0]
                    y_vals = data_to_plot[last_i:i,2]
                    if x_vals.shape[0]>1:
                        axis.plot(x_vals,y_vals,linewidth=linewidth, color = three_colors[last_daction])
                    else:
                        axis.scatter(x_vals,y_vals,s=40, color = three_colors[last_daction])
                    last_daction = new_daction
                    last_i = i
        elif line_style == "scatter_plot":
            dt = data_to_plot[-1,0] - data_to_plot[-2,0]
            #first I do the vertical dahsed line, so they get covered
            for i in range(data_to_plot.shape[0]-1):
                axis.plot([data_to_plot[i+1,0],data_to_plot[i+1,0]],[data_to_plot[i,2],data_to_plot[i+1,2]], color="lightgray",
                        linewidth=0.8)
            tri_act_data = data_to_plot[:,1]
            zero_mask = np.abs(tri_act_data) < 0.00001
            one_mask = np.abs(tri_act_data-1.) < 0.00001
            two_mask = np.abs(tri_act_data-2.) < 0.00001
            zero_data = data_to_plot[zero_mask]
            one_data = data_to_plot[one_mask]
            two_data = data_to_plot[two_mask]
            for i in range(2,data_to_plot.shape[1]):
                axis.scatter(zero_data[:,0],zero_data[:,i], color = next_color(color_iter),s=linewidth)
                axis.scatter(one_data[:,0],one_data[:,i], color = next_color(color_iter),s=linewidth)
                axis.scatter(two_data[:,0],two_data[:,i], color = next_color(color_iter),s=linewidth)
        else:
            print("Please choose a valid line_style")
    #main plotting if there is only a continuous action
    else:
        if line_style == "constant":
            color = next_color(color_iter)
            #compute the step to draw the length of the last point
            dt = data_to_plot[-1,0] - data_to_plot[-2,0]
            #first I do the vertical dahsed line, so they get covered
            if not hide_gray_vertical_line:
                for i in range(data_to_plot.shape[0]-1):
                    axis.plot([data_to_plot[i+1,0],data_to_plot[i+1,0]],[data_to_plot[i,1],data_to_plot[i+1,1]], color="lightgray",
                        linewidth=0.8)
            #horizontal lines
            data_to_plot = np.concatenate( (data_to_plot, np.array([[data_to_plot[-1,0]+dt,0.]]) ))
            for i in range(data_to_plot.shape[0]-1):  
                axis.plot([data_to_plot[i,0],data_to_plot[i+1,0]],[data_to_plot[i,1],data_to_plot[i,1]], color=color, linewidth=linewidth,
                        solid_capstyle = "butt")
        elif line_style == "scatter":
            for i in range(1,data_to_plot.shape[1]):
                axis.scatter(data_to_plot[:,0],data_to_plot[:,i], color = next_color(color_iter),s=linewidth)
        elif line_style == "plot":
            axis.plot(data_to_plot[:,0],data_to_plot[:,1], color = next_color(color_iter),linewidth=linewidth)
        elif line_style == "scatter_plot":
            dt = data_to_plot[-1,0] - data_to_plot[-2,0]
            #first I do the vertical dahsed line, so they get covered
            for i in range(data_to_plot.shape[0]-1):
                axis.plot([data_to_plot[i+1,0],data_to_plot[i+1,0]],[data_to_plot[i,1],data_to_plot[i+1,1]], color="lightgray",
                    linewidth=0.8)
            for i in range(1,data_to_plot.shape[1]):
                axis.scatter(data_to_plot[:,0],data_to_plot[:,i], color = next_color(color_iter),s=linewidth,zorder=100) 
        else:
            print("please choose a valid line_style")

    #set label details
    if not hide_yaxis_label:
        axis.set_ylabel(ylabel)
    if not hide_xaxis_label:    
        if xlabel != "":
            axis.set_xlabel(xlabel)
    if xticks is not None:
        axis.set_xticks(xticks)
    elif two_xticks:
        axis.set_xticks([ data_to_plot[0,0], data_to_plot[-1,0] ])
    if yticks is not None:
        axis.set_yticks(yticks)
    if k_notation:
        axis.xaxis.set_major_formatter(lambda x,y: num_to_k_notation(x) )
    if xticklabels is not None:
        axis.set_xticklabels(xticklabels)
    if yticklabels is not None:
        axis.set_yticklabels(yticklabels)

    #plot an extra function, if necessary
    if extra_cycles is not None:
        if not isinstance(extra_cycles[0], list) and not isinstance(extra_cycles[0], tuple):
            extra_cycles = [extra_cycles]
        #create the x points to plot
        for extra_cycle in extra_cycles:
            extra_func, extra_range, extra_color = extra_cycle
            x = np.linspace(extra_range[0],extra_range[1],250)
            y = [extra_func(x_val) for x_val in x]
            axis.plot(x,y, color=extra_color, linewidth=extra_cycles_linewidth, dashes=[4/extra_cycles_linewidth,2/extra_cycles_linewidth])
    
    #do the legend if necessary
    if legend_lines is not None:
        if legend_cols is None:
            legend_cols = len(legend_lines)
        axis.legend(legend_lines,legend_text,loc=legend_location,fancybox=False, framealpha=0.,borderaxespad=0.,
                    ncol=legend_cols,handlelength=legend_line_length,columnspacing=legend_column_spacing) #handlelength=subPlots[subIndex].legendLineLength,

def plot_sac_running_loss_on_axis(file_location, axis1, axis2, plot_to_file_line = None):
    plot_data = np.loadtxt(file_location).reshape(-1,3)
    if plot_to_file_line is not None:
        plot_data = plot_data[:plot_to_file_line+1]
    #plot q loss on first axis
    axis1.set_yscale("log")
    axis1.plot(plot_data[:,0],plot_data[:,1])
    axis1.set_xlabel("steps")
    axis1.set_ylabel("Q Running Loss")
    #plot pi loss on second axis
    axis2.plot(plot_data[:,0],plot_data[:,2])
    axis2.set_xlabel("steps")
    axis2.set_ylabel("Pi Running Loss")

def num_to_k_notation(tick, tex=True):
    """
    Used to produce ticks with a "k" indicating 1000

    Args:
        tick (float): value of the tick to be converted to a string
        tex (bool): if true, it will wrap the string in $$, making it look more "latex-like"

    Returns:
        tick_str (str): the string for the tick
    """
    tick_str = str(int(tick // 1000))
    end_not_stripped = str(int(tick % 1000))
    end = end_not_stripped.rstrip('0')
    if len(end) > 0:
        end = ("0"*(3-len(end_not_stripped))) + end
        tick_str += f".{end}"
    if tex:
        tick_str = "$" + tick_str + "$"
    tick_str += "k"
    return tick_str

def next_color(color_iterator):
    """
    It returns the next color if a non Non iterator is passed in. Otherwise
    None is returned

    Args:
        color_iterator(iterator): None or an iterator of colors
    """
    if color_iterator is None:
        return None
    else:
        return next(color_iterator)

def nearest_int(num):
    """ return the nearest integer to the float number num """
    return int(np.round(num))

def produce_otto_cycle(u_min,u_max,t1,t2,t3,t4,dt,t_range):
    """
    produces an extra_cycles for plot_actions_on_axis() or sac_paper_plot() representing
    an Otto cycle, i.e. a trapezoidal cycle. 

    Args:
        u_min (float): minimum value of u in the Otto cycle
        u_max (float): maximum value of u in the Otto cycle
        t1 (float): duration of the first stroke 
        t2 (float): duration of the second stroke
        t3 (float): duration of the third stroke
        t4 (float): duration of the forth stroke
        dt (float): duration of each time-step
        t_range (tuple(float)): time interval (ti, tf) where to plot the Otto cycle
    """
    t,t_fin = t_range
    t = t/dt
    t_fin = t_fin/dt
    t1 = t1/dt
    t2 = t2/dt
    t3 = t3/dt
    t4 = t4/dt
    times = [t1,t2,t3,t4]
    current_stroke = 0

    def otto_cycle(t,q_min,q_max,t1,t2,t3,t4):
        """ auxillary function describing the Otto cycle as a function of t"""
        t_tot = t1+t2+t3+t4
        #make it periodic setting it to first period
        t = t % t_tot
        if t< t1:
            return q_min
        elif t< t1+t2:
            return q_min + (q_max-q_min)*(t-t1)/t2
        elif t< t1+t2+t3:
            return q_max
        else:
            return q_max + (q_min-q_max)*(t-t1-t2-t3)/t4
    
    def stroke_color(stroke):
        """ auxillary function describing the color of the otto cycle """
        if stroke ==0:
            return "cornflowerblue"   
        elif stroke ==1:
            return "limegreen"
        elif stroke==2:
            return "orange"
        else:
            return "limegreen"

    extra_cycles = []
    while t < t_fin:
        new_t = min(t + times[current_stroke], t_fin)
        extra_cycles.append([lambda t,q_min=u_min,q_max=u_max,t1=t1,t2=t2,t3=t3,t4=t4: otto_cycle(t,q_min,q_max,t1,t2,t3,t4),
               [t,new_t],stroke_color(current_stroke)])
        t = new_t
        current_stroke = (current_stroke+1)%4
    return extra_cycles
 
def count_quantities(running_multi_obj_file):
    """
    counts the number of quantities in the multi_objective log. If it doesn't exist, 
    return 0.
    
    Args:
        running_multi_obj_file (str): file location

    Returns:
        quantities (int): number of quantities
    """
    if Path(running_multi_obj_file).exists():
        loaded_data = np.loadtxt(running_multi_obj_file)
        if len(loaded_data.shape) == 1:
            quantities = loaded_data.shape[0] - 1
        else:
            quantities = loaded_data.shape[1] - 1
    else:
        quantities = 0
    return quantities



