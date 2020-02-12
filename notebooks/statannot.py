'''
Main file from https://github.com/webermarcolivier/statannot.
Check out his awesome repo to add statistical annotations.
'''

import matplotlib.pyplot as plt
from matplotlib import lines
import matplotlib.transforms as mtransforms
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
import seaborn as sns
from seaborn.utils import remove_na

from scipy import stats

DEFAULT = object()


def stat_test(box_data1, box_data2, test, **stats_params):
    test_short_name = ''
    pval = None
    formatted_output = None
    if test == 'Levene':
        stat, pval = stats.levene(box_data1, box_data2, **stats_params)
        test_short_name = 'levene'
        formatted_output = ("Levene test of variance, "
                            "P_val={:.3e} stat={:.3e}").format(pval, stat)
    elif test == 'Mann-Whitney':
        u_stat, pval = stats.mannwhitneyu(
            box_data1, box_data2, alternative='two-sided', **stats_params)
        test_short_name = 'M.W.W.'
        formatted_output = ("Mann-Whitney-Wilcoxon test two-sided "
                            "P_val={:.3e} U_stat={:.3e}").format(pval, u_stat)
    elif test == 'Mann-Whitney-gt':
        u_stat, pval = stats.mannwhitneyu(
            box_data1, box_data2, alternative='greater', **stats_params)
        test_short_name = 'M.W.W.'
        formatted_output = ("Mann-Whitney-Wilcoxon test greater "
                            "P_val={:.3e} U_stat={:.3e}").format(pval, u_stat)
    elif test == 'Mann-Whitney-ls':
        u_stat, pval = stats.mannwhitneyu(
            box_data1, box_data2, alternative='less', **stats_params)
        test_short_name = 'M.W.W.'
        formatted_output = ("Mann-Whitney-Wilcoxon test smaller "
                            "P_val={:.3e} U_stat={:.3e}").format(pval, u_stat)
    elif test == 't-test_ind':
        stat, pval = stats.ttest_ind(a=box_data1, b=box_data2, **stats_params)
        test_short_name = 't-test_ind'
        formatted_output = ("t-test independent samples, "
                            "P_val={:.3e} stat={:.3e}").format(pval, stat)
    elif test == 't-test_welch':
        stat, pval = stats.ttest_ind(
            a=box_data1, b=box_data2, equal_var=False, **stats_params)
        test_short_name = 't-test_welch'
        formatted_output = ("Welch's t-test independent samples, "
                            "P_val={:.3e} stat={:.3e}").format(pval, stat)
    elif test == 't-test_paired':
        stat, pval = stats.ttest_rel(a=box_data1, b=box_data2, **stats_params)
        test_short_name = 't-test_rel'
        formatted_output = ("t-test paired samples, "
                            "P_val={:.3e} stat={:.3e}").format(pval, stat)
    elif test == 'Wilcoxon':
        if "zero_method" in stats_params.keys():
            zero_method = stats_params["zero_method"]
            del stats_params["zero_method"]
        else:
            zero_method = len(box_data1) <= 20 and "pratt" or "wilcox"
        print("Using zero_method ", zero_method)
        stat, pval = stats.wilcoxon(
            box_data1, box_data2, zero_method=zero_method, **stats_params)
        test_short_name = 'Wilcoxon'
        formatted_output = ("Wilcoxon test (paired samples), "
                            "P_val={:.3e} stat={:.3e}").format(pval, stat)
    elif test == 'Kruskal':
        stat, pval = stats.kruskal(box_data1, box_data2, **stats_params)
        test_short_name = 'Kruskal'
        formatted_output = ("Kruskal-Wallis paired samples, "
                            "P_val={:.3e} stat={:.3e}").format(pval, stat)
    return pval, formatted_output, test_short_name


def pval_annotation_text(x, pvalue_thresholds):
    single_value = False
    if type(x) is np.array:
        x1 = x
    else:
        x1 = np.array([x])
        single_value = True
    # Sort the threshold array
    pvalue_thresholds = pd.DataFrame(pvalue_thresholds).sort_values(by=0, ascending=False).values
    x_annot = pd.Series(["" for _ in range(len(x1))])
    for i in range(0, len(pvalue_thresholds)):
        if i < len(pvalue_thresholds)-1:
            condition = (x1 <= pvalue_thresholds[i][0]) & (pvalue_thresholds[i+1][0] < x1)
            x_annot[condition] = pvalue_thresholds[i][1]
        else:
            condition = x1 < pvalue_thresholds[i][0]
            x_annot[condition] = pvalue_thresholds[i][1]

    return x_annot if not single_value else x_annot.iloc[0]


def simple_text(pval, pvalue_format, pvalue_thresholds, test_short_name=None):
    """
    Generates simple text for test name and pvalue
    :param pval: pvalue
    :param pvalue_format: format string for pvalue
    :param test_short_name: Short name of test to show
    :param pvalue_thresholds: String to display per pvalue range
    :return: simple annotation
    """
    # Sort thresholds
    thresholds = sorted(pvalue_thresholds, key=lambda x: x[0])

    # Test name if passed
    text = test_short_name and test_short_name + " " or ""

    for threshold in thresholds:
        if pval < threshold[0]:
            pval_text = "p ≤ {}".format(threshold[1])
            break
    else:
        pval_text = "p = {}".format(pvalue_format).format(pval)

    return text + pval_text


def add_stat_annotation(ax,
                        data=None, x=None, y=None, hue=None, order=None,
                        hue_order=None, box_pairs=None, test='t-test_welch',
                        text_format='star', pvalue_format_string=DEFAULT,
                        text_annot_custom=None,
                        loc='inside', show_test_name=True,
                        pvalue_thresholds=DEFAULT, stats_params=dict(),
                        use_fixed_offset=False, line_offset_to_box=None,
                        line_offset=None, line_height=0.02, text_offset=1,
                        stack=True, color='0.2', linewidth=1.5,
                        fontsize='medium', verbose=1):
    """
    Computes statistical test between pairs of data series and add statistical annotation on top
    of the boxes. Uses the same exact arguments `data`, `x`, `y`, `hue`, `order`,
    `hue_order` as the seaborn boxplot function.

    :param line_height: in axes fraction coordinates
    :param text_offset: in points
    :param box_pairs: can be of either form: For non-grouped boxplot: `[(cat1, cat2), (cat3, cat4)]`. For boxplot grouped by hue: `[((cat1, hue1), (cat2, hue2)), ((cat3, hue3), (cat4, hue4))]`
    :param pvalue_format_string: defaults to `"{.3e}"`
    :param pvalue_thresholds: list of lists, or tuples. Default is: For "star" text_format: `[[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, "ns"]]`. For "simple" text_format : `[[1e-5, "1e-5"], [1e-4, "1e-4"], [1e-3, "0.001"], [1e-2, "0.01"]]`
    """

    def find_x_position_box(box_plotter, boxName):
        """
        boxName can be either a name "cat" or a tuple ("cat", "hue")
        """
        if box_plotter.plot_hues is None:
            cat = boxName
            hue_offset = 0
        else:
            cat = boxName[0]
            hue = boxName[1]
            hue_offset = box_plotter.hue_offsets[
                box_plotter.hue_names.index(hue)]

        group_pos = box_plotter.group_names.index(cat)
        box_pos = group_pos + hue_offset
        return box_pos

    def get_box_data(box_plotter, boxName):
        """
        boxName can be either a name "cat" or a tuple ("cat", "hue")

        Here we really have to duplicate seaborn code, because there is not
        direct access to the box_data in the BoxPlotter class.
        """
        cat = box_plotter.plot_hues is None and boxName or boxName[0]

        index = box_plotter.group_names.index(cat)
        group_data = box_plotter.plot_data[index]

        if box_plotter.plot_hues is None:
            # Draw a single box or a set of boxes
            # with a single level of grouping
            box_data = remove_na(group_data)
        else:
            hue_level = boxName[1]
            hue_mask = box_plotter.plot_hues[index] == hue_level
            box_data = remove_na(group_data[hue_mask])

        return box_data

    # Set default values if necessary
    if pvalue_format_string is DEFAULT:
        pvalue_format_string = '{:.3e}'
        simple_format_string = '{:.2f}'
    else:
        simple_format_string = pvalue_format_string

    if pvalue_thresholds is DEFAULT:
        if text_format == "star":
            pvalue_thresholds = [[1e-4, "****"], [1e-3, "***"],
                                 [1e-2, "**"], [0.05, "*"], [1, "ns"]]
        else:
            pvalue_thresholds = [[1e-5, "1e-5"], [1e-4, "1e-4"],
                                 [1e-3, "0.001"], [1e-2, "0.01"]]

    fig = plt.gcf()

    valid_list = ['inside', 'outside']
    if loc not in valid_list:
        raise ValueError("loc value should be one of the following: {}."
                         .format(', '.join(valid_list)))
    valid_list = ['full', 'simple', 'star']
    if text_format not in valid_list:
        raise ValueError("text_format value should be one of the following: {}."
                         .format(', '.join(valid_list)))
    valid_list = ['t-test_ind', 't-test_welch', 't-test_paired',
                  'Mann-Whitney', 'Mann-Whitney-gt', 'Mann-Whitney-ls',
                  'Levene', 'Wilcoxon', 'Kruskal']
    if test not in valid_list:
        raise ValueError("test value should be one of the following: {}."
                         .format(', '.join(valid_list)))

    if verbose >= 1 and text_format == 'star':
        print("pvalue annotation legend:")
        pvalue_thresholds = pd.DataFrame(pvalue_thresholds).sort_values(by=0, ascending=False).values
        for i in range(0, len(pvalue_thresholds)):
            if i < len(pvalue_thresholds)-1:
                print('{}: {:.2e} < p <= {:.2e}'.format(pvalue_thresholds[i][1],
                                                        pvalue_thresholds[i+1][0],
                                                        pvalue_thresholds[i][0]))
            else:
                print('{}: p <= {:.2e}'.format(pvalue_thresholds[i][1], pvalue_thresholds[i][0]))
        print()

    # Create the same BoxPlotter object as seaborn's boxplot
    box_plotter = sns.categorical._BoxPlotter(
        x, y, hue, data, order, hue_order, orient=None, width=.8, color=None,
        palette=None, saturation=.75, dodge=True, fliersize=5, linewidth=None)

    ylim = ax.get_ylim()
    yrange = ylim[1] - ylim[0]

    if line_offset is None:
        if loc == 'inside':
            line_offset = 0.05
            if line_offset_to_box is None:
                line_offset_to_box = 0.06
        # 'outside', see valid_list
        else:
            line_offset = 0.03
            if line_offset_to_box is None:
                line_offset_to_box = line_offset
    else:
        if loc == 'inside':
            if line_offset_to_box is None:
                line_offset_to_box = 0.06
        # 'outside', see valid_list
        else:
            line_offset_to_box = line_offset
    y_offset = line_offset*yrange
    y_offset_to_box = line_offset_to_box*yrange

    y_stack = []
    ann_list = []
    test_result_list = []

    for box1, box2 in box_pairs:
        group_names = box_plotter.group_names
        hue_names = box_plotter.hue_names
        if box_plotter.plot_hues is None:
            cat1 = box1
            cat2 = box2
            label1 = '{}'.format(cat1)
            label2 = '{}'.format(cat2)
            valid = cat1 in group_names and cat2 in group_names
        else:
            cat1 = box1[0]
            hue1 = box1[1]
            cat2 = box2[0]
            hue2 = box2[1]
            label1 = '{}_{}'.format(cat1, hue1)
            label2 = '{}_{}'.format(cat2, hue2)
            valid = (cat1 in group_names and cat2 in group_names and
                     hue1 in hue_names and hue2 in hue_names)

        if valid:
            # Get position of boxes
            x1 = find_x_position_box(box_plotter, box1)
            x2 = find_x_position_box(box_plotter, box2)
            box_data1 = get_box_data(box_plotter, box1)
            box_data2 = get_box_data(box_plotter, box2)
            ymax1 = box_data1.max()
            ymax2 = box_data2.max()

            pval, formatted_output, test_short_name = stat_test(box_data1, box_data2, test, **stats_params)
            test_result_list.append({'pvalue': pval, 'test_short_name': test_short_name,
                                     'formatted_output': formatted_output, 'box1': box1,
                                     'box2': box2})
            if verbose >= 1:
                print("{} v.s. {}: {}".format(label1, label2, formatted_output))

            if text_format == 'full':
                text = "{} p = {}".format('{}', pvalue_format_string).format(test_short_name, pval)
            elif text_format is None:
                text = None
            elif text_format is 'star':
                text = pval_annotation_text(pval, pvalue_thresholds)
            # 'simple', see valid_list
            else:
                test_short_name = show_test_name and test_short_name or ""
                text = simple_text(pval, simple_format_string, pvalue_thresholds, test_short_name)

            if loc == 'inside':
                yref = max(ymax1, ymax2)
            # 'outside', see valid_list
            else:
                yref = ylim[1]

            if stack:
                if len(y_stack) > 0:
                    yref2 = max(yref, max(y_stack))
                else:
                    yref2 = yref
            else:
                yref2 = yref

            if len(y_stack) == 0:
                y = yref2 + y_offset_to_box
            else:
                y = yref2 + y_offset
            h = line_height*yrange
            line_x, line_y = [x1, x1, x2, x2], [y, y + h, y + h, y]
            if loc == 'inside':
                ax.plot(line_x, line_y, lw=linewidth, c=color)
            # 'outside', see valid_list
            else:
                line = lines.Line2D(
                    line_x, line_y, lw=linewidth, c=color,
                    transform=ax.transData)
                line.set_clip_on(False)
                ax.add_line(line)

            ax.set_ylim((ylim[0], 1.1*(y + h)))

            if text is not None:
                ann = ax.annotate(
                    text, xy=(np.mean([x1, x2]), y + h),
                    xytext=(0, text_offset), textcoords='offset points',
                    xycoords='data', ha='center', va='bottom',
                    fontsize=fontsize, clip_on=False, annotation_clip=False)
                ann_list.append(ann)

                plt.draw()
                y_top_annot = None
                got_mpl_error = False
                if not use_fixed_offset:
                    try:
                        bbox = ann.get_window_extent()
                        bbox_data = bbox.transformed(ax.transData.inverted())
                        y_top_annot = bbox_data.ymax
                    except RuntimeError:
                        got_mpl_error = True

                if use_fixed_offset or got_mpl_error:
                    if verbose >= 1:
                        print("Warning: cannot get the text bounding box. "
                              "Falling back to a fixed y offset. "
                              "Layout may be not optimal.")
                    # We will apply a fixed offset in points,
                    # based on the font size of the annotation.
                    fontsize_points = FontProperties(size='medium').get_size_in_points()
                    offset_trans = mtransforms.offset_copy(
                        ax.transData, fig=fig, x=0,
                        y=1.0*fontsize_points + text_offset, units='points')
                    y_top_display = offset_trans.transform((0, y + h))
                    y_top_annot = ax.transData.inverted().transform(y_top_display)[1]
            else:
                y_top_annot = y + h

            y_stack.append(y_top_annot)
        else:
            raise ValueError("box_pairs contains an invalid box pair.")
            pass

    y_stack_max = max(y_stack)
    if loc == 'inside':
        ax.set_ylim((ylim[0], 1.03*y_stack_max))
    elif loc == 'outside':
        ax.set_ylim((ylim[0], ylim[1]))

    return ax, test_result_list
