# Annotatation of protein structures with attention weights
import tensorflow as tf
import logomaker as lm
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import Write_Results
from Bio.PDB import PDBParser
from dataset_creation import Conformational_Structure_Dataset
import re

aa_dict = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS",
           "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
           "Y": "TYR", "V": "VAL"}


# Attention distribution logo creation facilitator (Function incorporated into model source code)
def create_attention_dist_logo(attention_dict, primary_structure, num_decoder_layers, num_heads, vocab, epoch,
                               write_directory, type):
    # Average across attention heads per decoder layer
    if type == "PB":
        vocabulary_2 = vocab.copy()
        vocabulary_2["j"] = vocabulary_2.pop("<OTHER>")
        vocabulary_2.pop("<SOS>")
        vocabulary_2.pop("<EOS>")
        vocabulary_2.pop("<PAD>")
    else:
        vocabulary_2 = vocab.copy()

    for layer_number in range(1, num_decoder_layers + 1):
        attention_heads = tf.squeeze(attention_dict['decoder_layer' + str(layer_number) + '_block2'], 0)
        sum_heads = attention_heads[0]
        for att_head in attention_heads[1:]:
            sum_heads += att_head
        avg_heads = tf.math.divide(sum_heads, [num_heads])
        # Column-wise average (the average for each token in the antigen primary structure)
        avg_x = tf.math.reduce_mean(avg_heads, axis=0)

        # <SOS>, <EOS>, and <PAD> are omitted
        avg_x_numpy = avg_x.numpy()[1:-1]
        primary_structure_len = len(primary_structure)
        avg_x_numpy = avg_x_numpy[:primary_structure_len]
        create_logo_attention_dist(primary_structure, avg_x_numpy, layer_number, vocabulary_2, epoch, write_directory, type)

        if layer_number == 1:
            avg_all_layers = avg_x
        elif layer_number == num_decoder_layers:
            avg_all_layers += avg_x
            avg_all_layers = tf.math.divide(avg_all_layers, [num_decoder_layers])
            avg_all_layers = avg_all_layers.numpy()[1:-1]
            avg_all_layers = avg_all_layers[:primary_structure_len]
            create_logo_attention_dist(primary_structure, avg_all_layers, "all", vocabulary_2, epoch, write_directory, type)
        else:
            # element-wise addition of the averaged attention tensors across all decoder layers
            avg_all_layers += avg_x


# Creation of attention distribution logo (this does NOT need to be imported into model source code)
def create_logo_attention_dist(primary_structure, attention_dist, layer_number, vocab, epoch, write_directory, type):
    vocabulary_upper = []
    if type == "PB":
        for token in vocab:
            vocabulary_upper.append(token.upper())
    else:
        for token in vocab:
            vocabulary_upper.append(token.upper())
            vocabulary_upper = vocabulary_upper[2:]
    primary_structure_list = [aa.upper() for aa in primary_structure]
    df_antigen = pd.DataFrame(columns=vocabulary_upper, dtype=int)
    df_antigen.append(pd.Series(name="att"))
    att_list = attention_dist.tolist()
    for i, att in enumerate(att_list):
        df_antigen.loc[i, primary_structure_list[i]] = float(att)
    df_antigen.fillna(0, inplace=True)

    plt.figure()
    logo = lm.Logo(df_antigen, font_name='Arial Rounded MT Bold', color_scheme="skylign_protein", figsize=(45, 5))
    logo.ax.set_xlabel('Position', fontsize=14)
    logo.ax.set_ylabel("Probability", labelpad=1, fontsize=14)
    Write_Results.write_image(logo.fig, "d_" + str(layer_number) + "_logo_attn_dist", write_directory, epoch)


# Attention Node-edge graph creation facilitator (Function incorporated into model source code)
def create_node_edge_att(attention_dict, antigen_primary, predicted_primary, num_decoder_layers, num_heads,
                         epoch, write_directory):
    antigen_pdb_source_id = "7jmw"

    # Retrieving average attention scores over all heads per layer
    for layer_number in range(1, num_decoder_layers + 1):
        attention_heads = tf.squeeze(attention_dict['decoder_layer' + str(layer_number) + '_block2'], 0)
        sum_heads = attention_heads[0]
        for att_head in attention_heads[1:]:
            sum_heads += att_head
        avg_heads = tf.math.divide(sum_heads, [num_heads])

        if layer_number == 1:
            avg_all_layers = avg_heads
        else:
            avg_all_layers += avg_heads

    avg_all_layers = tf.math.divide(avg_all_layers, [num_decoder_layers])
    avg_all_layers = avg_all_layers.numpy()[:, 1:-1]  # SOS and EOS tokens are omitted
    primary_structure_len = len(antigen_primary)
    avg_all_layers = avg_all_layers[:, :primary_structure_len]  # PAD tokens are omitted
    create_node_edge_att_2(avg_all_layers, antigen_primary, predicted_primary, "all", epoch,
                           write_directory)


# Attention Node-edge graph creation (function does NOT need to be imported into model source code)
def create_node_edge_att_2(att_avg, antigen_primary, predicted_primary, layer_number, epoch, write_directory):
    # Dataframe alligning the attention score for each Fv AA with each antigen (A) AA (rows = fv_len * antigen_len)
    att_score_df = pd.DataFrame(columns=("fv_aa", "a_aa", "att_score", "fv_pos", "a_pos"))
    # Dataframe specifying the nodes and edges for attention
    node_df = pd.DataFrame(
        columns=("name", "tar", "att_score", "ResType", "ResIndex", "ResidueLabel", "ResChain", "SS", "interaction"))
    # Dataframe specifying the nodes and edges for the backbone of the variable domain primary structures
    variable_backbone_df = pd.DataFrame(
        columns=("name", "tar", "att_score", "ResType", "ResIndex", "ResidueLabel", "ResChain", "SS", "interaction"))

    # Identifying the end of the predicted variable domains - "J" predicted padding tokens are excluded
    predicted_primary = predicted_primary.lower()
    heavy_domain_end = predicted_primary.find("j")
    light_domain_end = predicted_primary[142:].find("j")
    light_domain_end = light_domain_end + 142

    # Alligning attention scores
    counter_y = -1
    counter_row = 0
    fv_pos = 0
    for fv_aa in predicted_primary:
        counter_y += 1
        counter_x = 0
        fv_pos += 1
        a_pos = 1  # Resets to 1 after iterating over one row of the antigen attention score (for one fv aa)
        for a_aa in antigen_primary:
            att_score_df.loc[counter_row] = [fv_aa, a_aa, att_avg[counter_y, counter_x], fv_pos, a_pos]
            counter_row += 1
            counter_x += 1
            a_pos += 1

    # Creating node-edge backbone dataframe for predicted variable domains - J tokens are not included
    # Adding heavy backbone
    counter_row = 0
    for row in att_score_df.itertuples():
        if row[0] == 0:
            first_element = True
        else:
            first_element = False
        if row[4] == heavy_domain_end + 1:     # The end of the heavy chain has been reached
            break

        if first_element:  # If its the first row, there is no previous row to index/validate
            pass
        else:
            if row[4] == att_score_df.loc[row[0] - 1][3]:  # Only one row per fv_AA AA (not per A AA)
                continue
        current_aa = aa_dict[row[1].upper()]
        current_pos = counter_row + 1
        current_node_name = current_aa + "_" + str(current_pos) + "_H"
        name = current_node_name
        att_score = row[3]
        restype = current_aa
        res_index = current_pos
        residue_label = current_aa + " " + str(current_pos) + " H"
        reschain = "H"
        interaction = "backbone_H"

        if row[4] != heavy_domain_end:    # Not equal to the second to last AA in the Fv primary structure
            next_pos = counter_row + 2
            next_aa_df = att_score_df[att_score_df["fv_pos"] == next_pos]
            next_aa = next_aa_df.iloc[0][0]
            next_aa = aa_dict[next_aa.upper()]
            tar = next_aa + "_" + str(next_pos) + "_H"
        else:
            tar = ""

        # locations of potential CDR residues are flagged with "SS" attribute
        if row[4] in range(91, 134):
            SS = "CDR_H3"
        elif row[4] in range(26, 36):
            SS = "CDR_H1"
        elif row[4] in range(51, 58):
            SS = "CDR H2"
        else:
            SS = ""

        variable_backbone_df.loc[counter_row] = [name, tar, att_score, restype, res_index, residue_label, reschain,
                                                 SS, interaction]
        counter_row += 1

    # Adding light backbone
    counter_row = 142  # For iterating over att_score_df and skipping the heavy chain + J tokens
    light_pos_index = 0  # With respect to pos in primary structure (this is for for graph attributes, NOT indexing df)
    df_new_index = variable_backbone_df.index[-1] + 1  # Indexing for creation of new rows in backbone df
    for row in att_score_df.itertuples():
        if row[4] < 142 + 1:    # Skipping past amino acids in heavy chain
            continue
        elif row[4] == light_domain_end + 1:     # The end of the light chain has been reached
            break
        elif row[4] == att_score_df.loc[row[0] - 1][3]:   # Only one row per fv_AA (not per A_AA)
            continue
        else:
            current_aa = aa_dict[row[1].upper()]
            current_pos = light_pos_index + 1
            current_node_name = current_aa + "_" + str(current_pos) + "_L"
            name = current_node_name
            att_score = row[3]
            restype = current_aa
            res_index = current_pos
            residue_label = current_aa + " " + str(current_pos) + " L"
            reschain = "L"
            interaction = "backbone_L"

            if row[4] != light_domain_end:
                next_pos_attn_df = current_pos + 142 + 1   # Next position as it appears on attn_df
                next_aa_df = att_score_df[att_score_df["fv_pos"] == next_pos_attn_df]
                next_aa = next_aa_df.iloc[0][0]
                next_aa = aa_dict[next_aa.upper()]
                next_pos = current_pos + 1
                tar = next_aa + "_" + str(next_pos) + "_L"
            else:
                tar = ""

            # locations of potential CDR residues are flagged with "SS" attribute
            if row[4] - 142 in range(24, 35):
                SS = "CDR_L1"
            elif row[4] - 142 in range(50, 57):
                SS = "CDR_L2"
            elif row[4] - 142 in range(89, 98):
                SS = "CDR L3"
            else:
                SS = ""

            variable_backbone_df.loc[df_new_index] = [name, tar, att_score, restype, res_index, residue_label, reschain,
                                                      SS, interaction]
            counter_row += 1
            df_new_index += 1
            light_pos_index += 1

    # rows specifying "J" padding tokens are removed from the attention score dataframe
    df_new_index = 0
    for row in att_score_df.itertuples():
        a_aa = aa_dict[row[2].upper()]
        att_score = row[3]
        fv_pos = row[4]
        a_pos = row[5]
        fv_aa = row[1]
        if fv_pos < heavy_domain_end + 1:  # +1 to offset python indexing convention
            fv_aa = aa_dict[row[1].upper()]  # Must be in loop, otherwise "J" token will cause error
            reschain = "H"
        elif light_domain_end + 1 > fv_pos > 141 + 1:  # +1 to both sides to offset python indexing convention
            fv_aa = aa_dict[row[1].upper()]
            reschain = "L"
            fv_pos = fv_pos - 142
        else:
            continue

        a_pos = a_pos + 333  # Position with respect to the full SARS-CoV-2 spike protein
        name = fv_aa + "_" + str(fv_pos) + "_" + reschain
        tar = "7jmw_modified.pdb#" + a_aa.capitalize() + " " + str(a_pos) + ".A"
        restype = fv_aa
        res_index = fv_pos
        res_label = fv_aa + " " + str(fv_pos) + " " + reschain
        SS = ""
        interaction = "attending"

        node_df.loc[df_new_index] = [name, tar, att_score, restype, res_index, res_label, reschain, SS, interaction]
        df_new_index += 1

    # Merging the two dataframes to create the final node table
    final_df = node_df.append(variable_backbone_df)
    final_df.to_csv(write_directory + "\\" + "epoch_" + str(epoch) + "_" + str(layer_number), header=True)
