import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, DataStructs

# EC plots utilities

def plot_ec_stratified_pie_chart(ec_numbers, min_angle_to_plot=5, depth=2, label_radii=None, label_font_sizes=None):
    """
    Plots a nested pie chart based on EC numbers.

    Parameters:
        ec_numbers (tuple): A tuple of EC numbers as strings (e.g., "2.1.1.1", "1.1.1.1").
        min_angle_to_plot (float): Minimum opening angle (in degrees) to display labels.
        depth (int): Number of hierarchical levels to include in the pie chart (1, 2, or 3).
        label_radii (list of float): Custom radii for label placement for each pie layer.
    """
    # Step 1: Organize EC numbers into hierarchical structure
    hierarchy = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for ec in ec_numbers:
        parts = ec.split(".")
        top_level = parts[0]
        sub_level = f"{parts[0]}.{parts[1]}" if depth > 1 and len(parts) > 1 else None
        sub_sub_level = f"{parts[0]}.{parts[1]}.{parts[2]}" if depth > 2 and len(parts) > 2 else None

        if depth == 1:
            hierarchy[top_level][None][None] += 1
        elif depth == 2:
            hierarchy[top_level][sub_level][None] += 1
        elif depth == 3:
            hierarchy[top_level][sub_level][sub_sub_level] += 1

    # Step 2: Prepare data for plotting
    innermost_labels = list(hierarchy.keys())
    innermost_sizes = [sum(sum(sub.values()) for sub in sub_levels.values()) for sub_levels in hierarchy.values()]

    inner_labels = []
    inner_sizes = []
    inner_colors_map = {}
    for top_level, sub_levels in hierarchy.items():
        if depth > 1:
            top_level_color = plt.cm.tab20b(np.linspace(0, 1, len(innermost_labels)))[list(hierarchy.keys()).index(top_level)]
            # print(plt.cm.tab20b(np.linspace(0, 1, len(innermost_labels))))
            # print(innermost_labels, top_level_color)
            for sub_level, sub_sub_levels in sub_levels.items():
                if sub_level:
                    inner_labels.append(sub_level)
                    inner_sizes.append(sum(sub_sub_levels.values()))
                    inner_colors_map[sub_level] = top_level_color

    outer_labels = []
    outer_sizes = []
    outer_colors_map = {}
    if depth > 2:
        for top_level, sub_levels in hierarchy.items():
            for sub_level, sub_sub_levels in sub_levels.items():
                for sub_sub_level, count in sub_sub_levels.items():
                    if sub_sub_level:
                        outer_labels.append(sub_sub_level)
                        outer_sizes.append(count)
                        outer_colors_map[sub_sub_level] = inner_colors_map[sub_level]

    # Define colors
    innermost_colors = plt.cm.tab20b(np.linspace(0, 1, len(innermost_labels)))
    inner_colors = [inner_colors_map[label] for label in inner_labels]
    outer_colors = [outer_colors_map[label] for label in outer_labels]

    # Default label radii if not provided
    if label_radii is None:
        label_radii = [0.5, 0.75, 1]
    if label_font_sizes is None:
        label_font_sizes = [10, 8, 6]  # Default font sizes for each layer

    # Step 3: Create the nested pie chart
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'aspect': 'equal'})

    # Innermost pie
    wedges_innermost, texts_innermost = ax.pie(innermost_sizes, radius=0.4, labels=innermost_labels,
                                               colors=innermost_colors, 
                                               wedgeprops=dict(width=0.35, edgecolor='w'))

    for label, wedge in zip(texts_innermost, wedges_innermost):
        theta = (wedge.theta1 + wedge.theta2) / 2  # Midpoint angle
        x, y = np.cos(np.radians(theta)), np.sin(np.radians(theta))
        label.set_position((x * label_radii[0], y * label_radii[0]))
        label.set_horizontalalignment("center")
        label.set_fontsize(label_font_sizes[0])



    # Adjust labels for inner pie based on angle
    for wedge, label in zip(wedges_innermost, texts_innermost):
        angle = (wedge.theta2 - wedge.theta1)
        if angle < min_angle_to_plot:
            label.set_text("")

    # Inner pie
    if depth > 1:
        wedges_inner, texts_inner = ax.pie(inner_sizes, radius=0.7, labels=inner_labels,
                                           colors=inner_colors, 
                                           wedgeprops=dict(width=0.29, edgecolor='w'))

        for label, wedge in zip(texts_inner, wedges_inner):
            theta = (wedge.theta1 + wedge.theta2) / 2
            x, y = np.cos(np.radians(theta)), np.sin(np.radians(theta))
            label.set_position((x * label_radii[1], y * label_radii[1]))
            label.set_horizontalalignment("center")
            label.set_fontsize(label_font_sizes[1])


        # Adjust labels for inner pie based on angle
        for wedge, label in zip(wedges_inner, texts_inner):
            angle = (wedge.theta2 - wedge.theta1)
            if angle < min_angle_to_plot:
                label.set_text("")

    # Outer pie
    if depth > 2:
        wedges_outer, texts_outer = ax.pie(outer_sizes, radius=1, labels=outer_labels,
                                           colors=outer_colors, 
                                           wedgeprops=dict(width=0.29, edgecolor='w'))

        for label, wedge in zip(texts_outer, wedges_outer):
            theta = (wedge.theta1 + wedge.theta2) / 2
            x, y = np.cos(np.radians(theta)), np.sin(np.radians(theta))
            label.set_position((x * label_radii[2], y * label_radii[2]))
            label.set_horizontalalignment("center")
            label.set_fontsize(label_font_sizes[2])


        # Adjust labels for outer pie based on angle
        for wedge, label in zip(wedges_outer, texts_outer):
            angle = (wedge.theta2 - wedge.theta1)
            if angle < min_angle_to_plot:
                label.set_text("")

    # Style the plot
    plt.setp(ax.texts,weight="bold", color='w')
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    plt.tight_layout()
    plt.savefig("pie_chart.pdf", format="pdf", bbox_inches="tight", pad_inches = 0)
    plt.show()



# IMPORTANT
def plot_accuracy_hist_kshot(N_experiments, N_experiments_class_accuracy_counts, test_hist_classes, test_size, model_name, y_lim, kshot = 0, valid_test_hist_classes = [], plot = True, trimmed = False):
    
    label_size = 25
    ticks_size = 20

    # Calculate accuracy per class based on test set frequency and N experiments
    updated_class_accuracy_counts = []

    # Assuming N_experiments_class_accuracy_counts is a list containing dictionaries of class accuracies for each experiment
    for i in range(N_experiments):
        accuracy_counts = N_experiments_class_accuracy_counts[i]
        if bool(valid_test_hist_classes):
            print(valid_test_hist_classes[i].values())
            fr_cl_values = valid_test_hist_classes[i].values()
            counts = [(x * 100 / fr_cl if fr_cl != 0 else 0) for x, fr_cl in zip(accuracy_counts.values(), fr_cl_values)]
            updated_class_accuracy_counts.append(counts)
        else:
            fr_cl_values = test_hist_classes.values()
            counts = [(x * 100 / fr_cl if fr_cl != 0 else 0) for x, fr_cl in zip(accuracy_counts.values(), fr_cl_values)]
            updated_class_accuracy_counts.append(counts)

    total_mean_accuracy = np.mean(updated_class_accuracy_counts, axis=1)
    mean_accuracies = np.mean(updated_class_accuracy_counts, axis=0)
    std_accuracies = np.std(updated_class_accuracy_counts, axis=0)

    total_std_accuracy = np.std(total_mean_accuracy)

    if plot:
        # Plot histograms with error bars, for distribution of test samples
        fig, ax1 = plt.subplots(figsize=(10, 7))
        color1 = 'coral'
        ax1.bar(np.arange(1, len(test_hist_classes)+1) -0.1, test_hist_classes.values(), width=0.4, color=color1, label='Distribution of test samples', align='center')
        ax1.set_xlabel('EC number class', size=label_size)
        ax1.set_ylabel('Counts per class in test set', color=color1, size=label_size)
        ax1.tick_params(axis='y', labelcolor=color1)

        # Create a twin y-axis for a second histogram, for accuracies per class with error bars
        ax2 = ax1.twinx()
        # color2 = 'royalblue'
        color2 = 'cornflowerblue'
        ax2.bar(np.arange(1, len(test_hist_classes)+1), mean_accuracies, width=0.4, color=color2, label='Accuracies per class', align='center', alpha=0.9)
        ax2.errorbar(np.arange(1, len(test_hist_classes)+1), mean_accuracies, yerr=std_accuracies, fmt='none', ecolor='black', capsize=5)
        ax2.set_ylabel('Model Accuracy (%)', color=color2, size=label_size)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim(0,y_lim*1.05)

        # Set thicker border for the plot
        for spine in ax1.spines.values():
            spine.set_linewidth(1.5)  # Adjust the thickness here

        # Increase the font size of the tick labels
        ax1.tick_params(axis='both', which='major', labelsize=ticks_size)
        ax2.tick_params(axis='both', which='major', labelsize=ticks_size)

        np.set_printoptions(precision=2)
        print()
        for i in range(N_experiments):
            formatted_accuracies = [f'{accuracy:.1f}' for accuracy in updated_class_accuracy_counts[i]]
            print(f'List of accuracies for experiment n.{i+1}: {formatted_accuracies}')
            # print(f'List of accuracies for experiment n.{i+1}: [{", ".join(f"{accuracy:.1f}" for accuracy in updated_class_accuracy_counts)}]')
        # Compute the average accuracy per entry across experiments
        average_per_entry = np.mean(updated_class_accuracy_counts, axis=0)
        formatted_avg_per_entry = [f'{avg:.1f}' for avg in average_per_entry]

        print(f'Average accuracy per entry across {N_experiments} experiments: {formatted_avg_per_entry}')

        print('List of global accuracies', total_mean_accuracy)
        print(f'Average global accuracy over {N_experiments} experiments: {np.mean(total_mean_accuracy):.1f} +- {total_std_accuracy:.1f}')
        print('test data counting distribution', test_hist_classes.values())

        
        # Adjust layout
        plt.tight_layout()
        plt.grid(alpha=0.25)
        plt.axhline(y=np.mean(total_mean_accuracy), color=color2, linestyle='-.', linewidth=1, label='Average accuracy')
        # plt.title(f'Distribution of {test_size} samples among enzyme classes, with average accuracy over {N_experiments} {kshot}-shot experiments with {model_name}')
        plt.legend(fontsize=15)

        plt.show()    

    return mean_accuracies, std_accuracies, total_mean_accuracy, total_std_accuracy



def plot_confusion_matrix_and_hist(conf_matrix, all_classes, actual_classes, level=1, hist = None, cmap=None):
    # Set the figure size and color palette

    fig = plt.figure(figsize=(16, 12))
    if hist != None:
        gs = gridspec.GridSpec(1, 3, width_ratios=[27, 0.5, 4])
    else:
        gs = gridspec.GridSpec(1, 2, width_ratios=[30, 1])

    # Create a subplot for the heatmap
    ax_heatmap = plt.subplot(gs[0])

    # Use a color palette
    color = 'Blues'
    if cmap is None:
        cmap = sns.color_palette(color, as_cmap=True)
    # cmap = cm

    # Create a heatmap with annotations
    if level == 1:
        sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap=cmap, cbar=False, 
                    linewidths=0.5, linecolor='gray', ax=ax_heatmap, annot_kws={'size':21})
    else:
        sns.heatmap(conf_matrix, annot=False, cmap=cmap, cbar=False, 
                    linewidths=0.5, linecolor='gray', ax=ax_heatmap)

    # Set axis labels
    ax_heatmap.set_xlabel('Predicted Class', fontsize=30)
    ax_heatmap.set_ylabel('Actual Class', fontsize=30)

    # Customize ticks and labels based on data range
    ax_heatmap.set_xticks(np.arange(len(all_classes)) + 0.5)
    ax_heatmap.set_xticklabels(all_classes, rotation=60, fontsize=24 - 5 * level)
    ax_heatmap.set_yticks(np.arange(len(actual_classes)) + 0.5)
    ax_heatmap.set_yticklabels(actual_classes, rotation=0, fontsize=24 - 5 * level)

    # Add a colorbar
    ax_colorbar = plt.subplot(gs[1])
    norm = plt.Normalize(vmin=np.min(conf_matrix), vmax=np.max(conf_matrix))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=ax_colorbar)
    ax_colorbar.tick_params(labelsize=20)

    if hist != None:
    
        bar_heights = [hist.get(actual_class, 0) for actual_class in actual_classes[::-1]]

        # Create a new subplot for the histogram on the right side
        ax_hist = plt.subplot(gs[2])

        # Plot the histogram with slightly narrower bars for spacing
        ax_hist.barh(np.arange(len(bar_heights)) + 0.5, bar_heights[::-1], 
                    color='black', alpha=0.8, height=0.96)  # Set height to <1 for spacing

        # Set x-axis to log scale and scientific notation
        ax_hist.set_xscale('log')
        # ax_hist.xaxis.set_major_formatter(LogFormatterSciNotation())

        # Set x-ticks
        ax_hist.set_xticks([1, 10, 100, 200])
        ax_hist.tick_params(axis='x', labelsize=20)  # Adjust size as needed

        # Set y-ticks to empty for histogram to avoid overlap with heatmap
        ax_hist.set_yticks(np.arange(len(actual_classes)) + 0.5)
        ax_hist.set_yticklabels([])

        # Align histogram y-limits with heatmap y-limits to ensure alignment
        ax_hist.set_ylim(ax_heatmap.get_ylim())
        ax_hist.grid(alpha=0.35)
        # Set title for histogram
        # ax_hist.set_title('Class Distribution', fontsize=20)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()



def plot_radar_chart(values_list, categories, colors, y_ticks, y_labels, y_min=0.0, y_max=1.0, 
                     label_radius1 = 1, label_radius2 = 1,title=None, legend_labels=None, legend_loc="upper right"):
    """
    Plots a radar chart for one or multiple sets of values.

    Args:
        values_list (list of list): Each inner list is a set of values (one per category).
        labels (list of str): Category labels (e.g., F1, Accuracy, etc.).
        categories (list of str): Categories to plot.
        colors (list): List of colors for each set of values.
        y_ticks (list): Ticks for the radial grid (e.g., [0.2, 0.4, 0.6, 0.8, 1.0]).
        y_labels (list): Labels corresponding to y_ticks.
        y_min (float): Minimum value for radial axis.
        y_max (float): Maximum value for radial axis.
        title (str): Plot title.
        legend_labels (list): Legend labels.
        legend_loc (str): Location of the legend.
    """
    # Number of variables
    num_vars = len(categories)

    # Compute angle of each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # close the circle

    # Plot setup
    fig, ax = plt.subplots(figsize=(13, 13), subplot_kw=dict(polar=True))

    # Plot each set of values
    for idx, values in enumerate(values_list):
        values = values + values[:1]  # close the circle
        ax.plot(angles, values, color=colors[idx], linewidth=2.5, label=legend_labels[idx] if legend_labels else None, alpha=0.8)

    # Set radial grid
    ax.set_rgrids(y_ticks, labels=y_labels, angle=45, fontsize=18, alpha=0.9)
    ax.set_ylim(y_min, y_max)

    # Hide default x-ticks
    ax.set_xticks([])
    ax.grid(linewidth=2.5, alpha=0.3)
    ax.spines['polar'].set_linewidth(2.2)

    # Manually add category labels
    for i, (angle, label) in enumerate(zip(angles[:-1], categories)):
        if i == 0 or i == 2:
            label_radius = y_max * label_radius1
        else:
            label_radius = y_max * label_radius2
        ax.text(angle, label_radius, label, fontsize=25, fontweight="bold", ha='center', va='center')

    # Add title and legend
    if title:
        plt.title(title, size=20, y=1.1)
    if legend_labels:
        plt.legend(loc=legend_loc, fontsize=20, bbox_to_anchor=(1.2, 1.15))

    plt.tight_layout()
    plt.show()



# Substrate/products plots utilities



def plot_pie_chart(data, total_len, max_radius=1.5, layer_width=0.08, palette='blue'):
    """
    Plot a multi-layer pie chart showing SMILES prediction evaluation breakdown.
    """
    import matplotlib.pyplot as plt

    total_valid = sum(data.values())
    wrong_gen = total_len - total_valid
    print(total_len, total_valid)

    outer_layer = [wrong_gen, data["invalid"], total_valid]
    inner_layer = [wrong_gen, data["invalid"], data["canonical_match"],
                   data["noncanonical_match"], data["canonical_valid"], data["noncanonical_valid"]]

    if palette == 'blue':
        outer_color = "midnightblue"
        inner_colors = list(plt.cm.Blues(np.linspace(0.75, 0.2, 4)))
    elif palette == 'green':
        outer_color = "darkgreen"
        inner_colors = list(plt.cm.Greens(np.linspace(0.75, 0.2, 4)))
    else:
        outer_color = "gray"
        inner_colors = list(plt.cm.Greys(np.linspace(0.75, 0.2, 4)))

    outer_colors = ["lightgrey", "tomato", outer_color]
    inner_colors = ["white", "white"] + inner_colors

    threshold = 2.0
    inner_labels = [
        "", "",  # for wrong_gen and invalid
        r"$\bf{CM}$" + f"\n{data['canonical_match']/total_valid*100:.1f}%" if data['canonical_match']/total_valid*100 >= threshold else "",
        r"$\bf{NCM}$" + f"\n{data['noncanonical_match']/total_valid*100:.1f}%" if data['noncanonical_match']/total_valid*100 >= threshold else "",
        r"$\bf{CV}$" + f"\n{data['canonical_valid']/total_valid*100:.1f}%" if data['canonical_valid']/total_valid*100 >= threshold else "",
        r"$\bf{NCV}$" + f"\n{data['noncanonical_valid']/total_valid*100:.1f}%" if data['noncanonical_valid']/total_valid*100 >= threshold else "",
    ]

    fig, ax = plt.subplots()
    ax.pie(outer_layer, radius=max_radius, colors=outer_colors, labeldistance=1.1,
           wedgeprops=dict(width=layer_width, edgecolor='white', linewidth=2), startangle=90)
    ax.pie(inner_layer, radius=max_radius - 1.2*layer_width, colors=inner_colors,
           labels=inner_labels, labeldistance=0.4,
           wedgeprops=dict(width=7*layer_width, edgecolor='white', linewidth=2), startangle=90,
           textprops={'fontsize': 11})

    ax.text(-0.25, 1.25, f"\n{data['invalid'] / total_valid * 100:.1f}%", ha='center', va='center', fontsize=11)

    fig.patch.set_facecolor('white')
    ax.set_facecolor('black')

    legend_labels = [
        "Wrong generation", "Invalid", "Valid", "",
        "", r"$\bf{CM}$: Canonical Match", r"$\bf{NCM}$: Non-Canonical Match",
        r"$\bf{CV}$: Canonical Valid", r"$\bf{NCV}$: Non-Canonical Valid"
    ]
    plt.legend(legend_labels, loc="center left", bbox_to_anchor=(1.1, 0.9), fontsize=11)
    plt.show()



def compute_tanimoto_similarity(smiles1, smiles2):
    """
    Compute Tanimoto similarity (basic + Morgan w/ chirality) between two SMILES.
    Returns a tuple of (basic_score, morgan_score).
    """
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return -1.0, -1.0

    fp1_basic = Chem.RDKFingerprint(mol1)
    fp2_basic = Chem.RDKFingerprint(mol2)
    tanimoto_basic = DataStructs.FingerprintSimilarity(fp1_basic, fp2_basic)

    fp1_morgan = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, useChirality=True)
    fp2_morgan = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, useChirality=True)
    tanimoto_morgan = DataStructs.FingerprintSimilarity(fp1_morgan, fp2_morgan)

    return tanimoto_basic, tanimoto_morgan



def average_tanimoto_values(tanimoto_lists, bins):
    """
    Compute an average histogram (with error bars) from multiple tanimoto value lists.
    Returns:
        frequencies: average frequency for each bin
        bin_edges: histogram bin edges
        raw_histograms: the raw count matrix per list
    """
    filtered_lists = [np.array(data)[np.array(data) != -1] for data in tanimoto_lists]
    all_data = np.concatenate(filtered_lists)
    _, bin_edges = np.histogram(all_data, bins=bins)

    histograms = []
    for data in filtered_lists:
        counts, _ = np.histogram(data, bins=bin_edges)
        histograms.append(counts)

    histograms = np.array(histograms)
    average_counts = np.mean(histograms, axis=0)
    avg_total = np.mean([len(data) for data in filtered_lists])
    frequencies = average_counts / avg_total

    return frequencies, bin_edges, histograms



def average_dict_list(dict_list, den=None):
    """Average values across a list of dictionaries. Optionally normalize by denominator."""
    summed = {}
    for d in dict_list:
        for k, v in d.items():
            summed[k] = summed.get(k, 0) + v
    avg = {k: summed[k] / len(dict_list) for k in summed}
    if den:
        percentages = {k: (v / den) * 100 for k, v in avg.items()}
        return avg, percentages
    return avg



def plot_average_tanimoto_histogram(tanimoto_lists, bins=30, color='blue'):
    """
    Plot an average histogram across multiple Tanimoto distributions.
    Includes standard deviation as error bars.
    """
    avg_counts, bin_edges, histograms = average_tanimoto_values(tanimoto_lists, bins)
    std_devs = np.std(histograms, axis=0) / np.mean([len(l) for l in tanimoto_lists if len(l) > 0])

    plt.figure(figsize=(10, 6))
    plt.bar(bin_edges[:-1], avg_counts * 100, width=np.diff(bin_edges), edgecolor="black",
            align="edge", alpha=0.7, linewidth=1.4, color=color)

    non_zero = avg_counts > 0
    plt.errorbar(bin_edges[:-1][non_zero] + np.diff(bin_edges)[non_zero] / 2,
                 avg_counts[non_zero] * 100, yerr=std_devs[non_zero] * 100,
                 fmt='none', color='black', capsize=3, label="Std Dev")
    
    # Calculate percentages for each category
    percentages_1_0 = [(sublist.count(1.0) / len(sublist)) * 100 if len(sublist) > 0 else 0 for sublist in tanimoto_lists]
    percentages_above_0_95_diff_1 = [(sum(1 for x in sublist if (x > 0.95 and x < 1.0)) / len(sublist)) * 100 if len(sublist) > 0 else 0 for sublist in tanimoto_lists]
    percentages_above_0_95 = [(sum(1 for x in sublist if x > 0.95) / len(sublist)) * 100 if len(sublist) > 0 else 0 for sublist in tanimoto_lists]
    percentages_above_0_85 = [(sum(1 for x in sublist if x > 0.85) / len(sublist)) * 100 if len(sublist) > 0 else 0 for sublist in tanimoto_lists]

    for s in tanimoto_lists:
        print(len(s))
    # Print the normalized percentages for each sublist
    print(f"Percentage of elements equal to 1.0: {percentages_1_0}")
    print(f"Percentage of elements above 0.95 but different from 1: {percentages_above_0_95_diff_1}")
    print(f"Percentage of elements above 0.95: {percentages_above_0_95}")
    print(f"Percentage of elements above 0.85: {percentages_above_0_85}")

    plt.xlabel("Tanimoto Similarity", size=22)
    plt.ylabel("Frequency (%)", size=22)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_smiles_grid_tight(smiles_matrix, legends_matrix=None, mol_size=(900, 400)):
    """
    Plots a tight grid of molecules with optional legends using RDKit.
    """
    mols_matrix = [[Chem.MolFromSmiles(s) for s in row] for row in smiles_matrix]
    img = Draw.MolsMatrixToGridImage(
        molsMatrix=mols_matrix,
        legendsMatrix=legends_matrix,
        subImgSize=mol_size,
    )
    return img



def analyze_tanimoto_of_branching(ground_truth_file: str, prod: bool = True):
    """
    Analyze Tanimoto similarity of branching alternatives in ground truth data.
    """
    with open(ground_truth_file, 'r') as f:
        ground_truth_data = json.load(f)

    ground_truth_smiles = [entry['raw_output'] for entry in ground_truth_data]
    other_truths = [entry['other_raw_outputs'] for entry in ground_truth_data]

    average_scores, std_devs, length_sublist = [], [], []
    scores = []
    seen = set()

    for sublist, ref in zip(other_truths, ground_truth_smiles):
        if sublist and tuple(sublist) not in seen:
            sub_scores = [compute_tanimoto_similarity(ref, s)[0] for s in sublist if s != ref]
            scores.extend(sub_scores)
            seen.add(tuple(sublist))

            avg_score = np.nanmean(sub_scores)
            std_dev = np.nanstd(sub_scores)
            average_scores.append(avg_score)
            std_devs.append(std_dev)
            length_sublist.append(len(sub_scores))

            if avg_score < 0.2:
                print(f"Low avg score: {avg_score:.2f} for {sublist}")

    # Plotting
    if prod:
        color_map = ['lightskyblue', 'dodgerblue', 'slateblue']
        xlabel = "Branching Products"
    else:
        color_map = ['darkseagreen', 'limegreen', 'gold']
        xlabel = "Branching Substrates"

    def assign_color(length):
        if length == 1:
            return color_map[0]
        elif 2 <= length < 5:
            return color_map[1]
        return color_map[2]

    colors = [assign_color(l) for l in length_sublist]

    plt.figure(figsize=(15, 6))
    for i in range(len(average_scores)):
        plt.errorbar(i, average_scores[i], yerr=std_devs[i], fmt='o', color=colors[i],
                     capsize=5, markersize=7, alpha=0.7)

    plt.xlabel(xlabel, fontsize=23)
    plt.ylabel('Average Tanimoto score', fontsize = 23)
    plt.gca().set_xticklabels([])
    plt.gca().set_xticks([])
    plt.yticks(fontsize = 15)
    plt.ylim(-0.01, 1.05)
    plt.grid(alpha = 0.3, axis='y')

    from matplotlib.patches import Patch
    legend = [
        Patch(color=color_map[0], label="Size = 2"),
        Patch(color=color_map[1], label="2 < Size < 5"),
        Patch(color=color_map[2], label="Size ≥ 5"),
    ]
    plt.legend(handles=legend, loc="lower right", fontsize=15)

    ax = plt.gca()
    for side in ax.spines.values():
        side.set_linewidth(1.5)

    plt.tight_layout()
    plt.show()



def plot_average_pie_chart(file_path, den, palette="blue"):
    with open(file_path, "r") as f:
        accuracy_data = json.load(f)
    print(accuracy_data)
    categories_average, _ = average_dict_list(accuracy_data, den=den)
    plot_pie_chart(categories_average, total_len=den, palette=palette)