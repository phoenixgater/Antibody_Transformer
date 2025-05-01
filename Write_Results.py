# Writes results, and saves images to a local directory
import os
import Query_SabDab
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# The directory in which results are to be written
root_directory = "D:\PHD_CODE\\Results"

# Used to determine the most successful design across multiple predictions
all_fv_results = {}
all_cdr_results = {}


# Assigns a local directory for the transformers written results
def assign_local_directory(schedule_identity):
    schedule_directory = root_directory + "\\" + "schedule " + schedule_identity
    p = Path(schedule_directory)
    if p.is_dir():
        pass
    else:
        os.mkdir(schedule_directory)

    directory_files = os.listdir(schedule_directory)
    taken_directories = []
    for file in directory_files:
        if file.find("Result") > -1:
            taken_directories.append(int(file.split()[1]))
    if len(taken_directories) == 0:
        result_number = 1
    else:
        result_number = (max(taken_directories)) + 1
    result_directory = schedule_directory + "\\" + "Result " + str(result_number)
    os.mkdir(result_directory)
    return result_directory


# Writes the transformer configuration and results to a txt file
def write_results(write_directory, written_results_dict, model_type="NMT", multiple_designs=False):
    text_file = open(write_directory + "\\" + "result.txt", "w")
    text_file.write("Model identity: " + written_results_dict["model_identity"] + "\n")
    text_file.write("TRANSFORMER CONFIGURATION:" + "\n")
    text_file.write("Number of encoder layers: " + written_results_dict["encoder_layers"] + "\n")
    text_file.write("Number of decoder layers: " + written_results_dict["decoder_layers"] + "\n")
    text_file.write("Embedding dimension (d_model): " + written_results_dict["embedding_dim"] + "\n")
    text_file.write("feed forward network width (ffn_width): " + written_results_dict["ffn_width"] + "\n")
    text_file.write("number of attention heads (in each multi-head attention layer): " +
                    written_results_dict["num_heads"] + "\n")
    text_file.write("Input vocab size: " + written_results_dict["input_vocab_size"] + "\n")
    text_file.write("Output vocab size: " + written_results_dict["target_vocab_size"] + "\n")
    text_file.write("Dropout rate: " + written_results_dict["dropout_rate"] + "\n")
    text_file.write("Number of Epochs: " + str(written_results_dict["Epochs"]) + "\n")
    text_file.write("Input sequence length: " + written_results_dict["input_seq_length"] + "\n")
    text_file.write("Output sequence length: " + written_results_dict["output_seq_length"] + "\n")
    text_file.write("\n")

    text_file.write("DATASET: " + "\n")
    text_file.write("Dataset identity: " + written_results_dict["dataset_identity"] + "\n")
    text_file.write("Number of training samples: " + written_results_dict["train_ds_size"] + "\n")
    text_file.write("Number of validation samples: " + written_results_dict["val_ds_size"] + "\n")
    text_file.write("Number of test samples: " + written_results_dict["test_ds_size"] + "\n")
    text_file.write("\n")

    text_file.write("METRICS: " + "\n")
    for i in range(len(written_results_dict["Epochs"])):
        text_file.write("Test accuracy at epoch " + str(written_results_dict["Epochs"][i]) + ": "
                        + written_results_dict["test_accuracy"][i] + "\n")
        text_file.write("Test loss at epoch " + str(written_results_dict["Epochs"][i]) + ": "
                        + written_results_dict["test_loss"][i] + "\n")
    text_file.write("\n")

    '''
    text_file.write("Predicted epitopes:" + "\n")
    text_file.write("With respect to antigen: ")
    for i in written_results_dict["epitopes_rbd"]:
        text_file.write(i + ", ")
    text_file.write("\n")
    text_file.write("With respect to full protein: ")
    for i in written_results_dict["epitopes_spike"]:
        text_file.write(i + ", ")
    text_file.write("\n")
    text_file.write("\n")
    '''

    text_file.write("PREDICTIONS:" + "\n")
    text_file.write("Antigen sequence 1 used: " + written_results_dict["antigen_sequence_1"] + "\n")
    if not multiple_designs:
        for i in range(len(written_results_dict["Epochs"])):
            text_file.write("PREDICTION AT EPOCH: " + str(written_results_dict["Epochs"][i]) + ": " + "\n")
            if model_type == "NMT":
                text_file.write("Predicted Fv heavy domain: " + written_results_dict["predicted_fv_heavy_1"][i] + "\n")
                text_file.write("Predicted Fv light domain: " + written_results_dict["predicted_fv_light_1"][i] + "\n")
                text_file.write("\n")
            else:
                text_file.write("Reconstructed antigen: " + written_results_dict["reconstructed_antigen"][i] + "\n")
                text_file.write("Sequence similarity: " + str(written_results_dict["sequence_similarity"][i]) + "\n")
                text_file.write("\n")

        if model_type == "NMT":
            text_file.write("SABDAB QUERY WHOLE FV: " + "\n")
            for i in range(len(written_results_dict["Epochs"])):
                text_file.write("Query results for design at Epoch " + str(written_results_dict["Epochs"][i]) + ":")
                text_file.write("\n")
                for pdb_id in written_results_dict["sabdab_query_whole_1"][i].keys():
                    text_file.write(pdb_id)
                    text_file.write(": " + str(written_results_dict["sabdab_query_whole_1"][i][pdb_id]))
                    text_file.write("\n")
                text_file.write("\n")

            text_file.write("SABDAB QUERY CDR ONLY: " + "\n")
            for i in range(len(written_results_dict["Epochs"])):
                text_file.write("Query results for design at Epoch " + str(written_results_dict["Epochs"][i]) + ":")
                text_file.write("\n")
                for pdb_id in written_results_dict["sabdab_query_cdr_1"][i].keys():
                    text_file.write(pdb_id)
                    text_file.write(": " + str(written_results_dict["sabdab_query_cdr_1"][i][pdb_id]))
                    text_file.write("\n")
                text_file.write("\n")

            text_file.write("SABDAB QUERY CDRH3 ONLY: " + "\n")
            for i in range(len(written_results_dict["Epochs"])):
                text_file.write("Query results for design at Epoch " + str(written_results_dict["Epochs"][i]) + ":")
                text_file.write("\n")
                for pdb_id in written_results_dict["sabdab_query_cdrh3_1"][i].keys():
                    text_file.write(pdb_id)
                    text_file.write(": " + str(written_results_dict["sabdab_query_cdrh3_1"][i][pdb_id]))
                    text_file.write("\n")
                text_file.write("\n")

    if multiple_designs:
        text_file.write("Antigen sequence 2 used: " + written_results_dict["antigen_sequence_2"] + "\n")
        for i in range(len(written_results_dict["Epochs"])):
            text_file.write("PREDICTION AT EPOCH: " + str(written_results_dict["Epochs"][i]) + ": " + "\n")
            text_file.write("Predicted Fv heavy domain_1: " + written_results_dict["predicted_fv_heavy_1"][i] + "\n")
            text_file.write("Predicted Fv light domain_1: " + written_results_dict["predicted_fv_light_1"][i] + "\n")
            text_file.write("Predicted Fv heavy domain_2: " + written_results_dict["predicted_fv_heavy_2"][i] + "\n")
            text_file.write("Predicted Fv light domain_2: " + written_results_dict["predicted_fv_light_2"][i] + "\n")
            text_file.write("\n")

        text_file.write("SABDAB QUERY WHOLE FV: " + "\n")
        for i in range(len(written_results_dict["Epochs"])):
            text_file.write("Query results for design 1 at Epoch " + str(written_results_dict["Epochs"][i]) + ":")
            text_file.write("\n")
            for pdb_id in written_results_dict["sabdab_query_whole_1"][i].keys():
                text_file.write(pdb_id)
                text_file.write(": " + str(written_results_dict["sabdab_query_whole_1"][i][pdb_id]))
                text_file.write("\n")

            text_file.write("\n")

            text_file.write("Query results for design 2 at Epoch " + str(written_results_dict["Epochs"][i]) + ":")
            text_file.write("\n")
            for pdb_id in written_results_dict["sabdab_query_whole_2"][i].keys():
                text_file.write(pdb_id)
                text_file.write(": " + str(written_results_dict["sabdab_query_whole_2"][i][pdb_id]))
                text_file.write("\n")

            text_file.write("\n")

        text_file.write("SABDAB QUERY CDR ONLY: " + "\n")
        for i in range(len(written_results_dict["Epochs"])):
            text_file.write("Query results for design 1 at Epoch " + str(written_results_dict["Epochs"][i]) + ":")
            text_file.write("\n")
            for pdb_id in written_results_dict["sabdab_query_cdr_1"][i].keys():
                text_file.write(pdb_id)
                text_file.write(": " + str(written_results_dict["sabdab_query_cdr_1"][i][pdb_id]))
                text_file.write("\n")

            text_file.write("\n")
            text_file.write("Query results for design 2 at Epoch " + str(written_results_dict["Epochs"][i]) + ":")
            text_file.write("\n")
            for pdb_id in written_results_dict["sabdab_query_cdr_2"][i].keys():
                text_file.write(pdb_id)
                text_file.write(": " + str(written_results_dict["sabdab_query_cdr_2"][i][pdb_id]))
                text_file.write("\n")
            text_file.write("\n")

        text_file.write("SABDAB QUERY CDRH3 ONLY: " + "\n")
        for i in range(len(written_results_dict["Epochs"])):
            text_file.write("Query results for design 1 at Epoch " + str(written_results_dict["Epochs"][i]) + ":")
            text_file.write("\n")
            for pdb_id in written_results_dict["sabdab_query_cdrh3_1"][i].keys():
                text_file.write(pdb_id)
                text_file.write(": " + str(written_results_dict["sabdab_query_cdrh3_1"][i][pdb_id]))
                text_file.write("\n")

            text_file.write("\n")
            text_file.write("Query results for design 2 at Epoch " + str(written_results_dict["Epochs"][i]) + ":")
            text_file.write("\n")
            for pdb_id in written_results_dict["sabdab_query_cdrh3_2"][i].keys():
                text_file.write(pdb_id)
                text_file.write(": " + str(written_results_dict["sabdab_query_cdrh3_2"][i][pdb_id]))
                text_file.write("\n")
            text_file.write("\n")


        # Overview writing temporarily disabled (requires adjustment to accomodate new models)

    write_results_overview(write_directory, written_results_dict)



# Writes images to the directory
def write_image(image, image_identity, write_directory, epoch="", layer_number=""):
    if image_identity == "attention_head":
        taken_identities = []
        directory_files = os.listdir(write_directory)
        for file in directory_files:
            if file.find("epoch_" + str(epoch) + "_" + "layer" + "_" + layer_number + "_" + "attention_head") > -1:
                taken_identities.append(int(file.split()[1][:-4]))
        if len(taken_identities) == 0:
            image.savefig(write_directory + "\\" + "epoch_" + str(
                epoch) + "_" + "layer" + "_" + layer_number + "_" + image_identity + " 1" + ".png")
        else:
            image.savefig(write_directory + "\\" + "epoch_" + str(
                epoch) + "_" + "layer" + "_" + layer_number + "_" + image_identity + " " +
                          str(max(taken_identities) + 1) + ".png")

    elif image_identity == "self_att_head_enc":
        taken_identities = []
        directory_files = os.listdir(write_directory)
        for file in directory_files:
            if file.find("epoch_" + str(epoch) + "_" + "layer" + "_" + layer_number + "_" + image_identity) > -1:
                taken_identities.append(int(file.split()[1][:-4]))
        if len(taken_identities) == 0:
            image.savefig(write_directory + "\\" + "epoch_" + str(
                epoch) + "_" + "layer" + "_" + layer_number + "_" + image_identity + " 1" + ".png")
        else:
            image.savefig(write_directory + "\\" + "epoch_" + str(
                epoch) + "_" + "layer" + "_" + layer_number + "_" + image_identity + " " +
                          str(max(taken_identities) + 1) + ".png")

    elif image_identity == "self_att_head_dec":
        taken_identities = []
        directory_files = os.listdir(write_directory)
        for file in directory_files:
            if file.find("epoch_" + str(epoch) + "_" + "layer" + "_" + layer_number + "_" + image_identity) > -1:
                taken_identities.append(int(file.split()[1][:-4]))
        if len(taken_identities) == 0:
            image.savefig(write_directory + "\\" + "epoch_" + str(
                epoch) + "_" + "layer" + "_" + layer_number + "_" + image_identity + " 1" + ".png")
        else:
            image.savefig(write_directory + "\\" + "epoch_" + str(
                epoch) + "_" + "layer" + "_" + layer_number + "_" + image_identity + " " +
                          str(max(taken_identities) + 1) + ".png")
    else:
        image.savefig(write_directory + "\\" + "epoch_" + str(epoch) + "_" + image_identity + ".png")


# Write results to the overview file
def write_results_overview(write_directory, written_results_dict):
    directory_files = os.listdir(root_directory)

    max_fv_results = []
    for i in range(len(written_results_dict["Epochs"])):
        identities = []

        for key in written_results_dict["sabdab_query_whole_1"][i].keys():
            identities.append(written_results_dict["sabdab_query_whole_1"][i][key][0])
        max_identity = max(identities)
        for key in written_results_dict["sabdab_query_whole_1"][i].keys():
            if written_results_dict["sabdab_query_whole_1"][i][key][0] == max_identity:
                max_result = []
                for result in written_results_dict["sabdab_query_whole_1"][i][key]:
                    max_result.append(result)
                max_result.append(written_results_dict["Epochs"][i])
                max_fv_results.append(max_result)
                break

    max_cdr_results = []
    for i in range(len(written_results_dict["Epochs"])):
        identities = []
        for key in written_results_dict["sabdab_query_cdr_1"][i].keys():
            identities.append(written_results_dict["sabdab_query_cdr_1"][i][key][0])
        max_identity = max(identities)
        for key in written_results_dict["sabdab_query_cdr_1"][i].keys():
            if written_results_dict["sabdab_query_cdr_1"][i][key][0] == max_identity:
                max_result = []
                for result in written_results_dict["sabdab_query_cdr_1"][i][key]:
                    max_result.append(result)
                max_result.append(written_results_dict["Epochs"][i])
                max_cdr_results.append(max_result)
                break

    max_cdrh3_results = []
    for i in range(len(written_results_dict["Epochs"])):
        identities = []
        for key in written_results_dict["sabdab_query_cdrh3_1"][i].keys():
            identities.append(written_results_dict["sabdab_query_cdrh3_1"][i][key][0])
        max_identity = max(identities)
        for key in written_results_dict["sabdab_query_cdrh3_1"][i].keys():
            if written_results_dict["sabdab_query_cdrh3_1"][i][key][0] == max_identity:
                max_result = []
                for result in written_results_dict["sabdab_query_cdrh3_1"][i][key]:
                    max_result.append(result)
                max_result.append(written_results_dict["Epochs"][i])
                max_cdrh3_results.append(max_result)
                break


    for i in range(len(written_results_dict["Epochs"])):
        if len(max_fv_results) > 4:
            epoch = max_fv_results[i][3]
        else:
            epoch = max_fv_results[i][2]
        max_identity_fv = max_fv_results[i][0]
        max_identity_name_fv = max_fv_results[i][1]
        max_identity_cdr = max_cdr_results[i][0]
        max_identity_name_cdr = max_cdr_results[i][1]
        max_identity_cdrh3 = max_cdrh3_results[i][0]
        max_identity_name_cdrh3 = max_cdrh3_results[i][1]

        written_content = {"schedule identity": written_results_dict["schedule identity"],
                           "result number": (os.path.split(write_directory)[1]).split()[1],
                           "epoch": epoch,
                           "highest Fv identity": max_identity_fv,
                           "highest Fv identity name": max_identity_name_fv,
                           "highest CDR identity": max_identity_cdr,
                           "highest CDR identity name": max_identity_name_cdr,
                           "highest CDR H3 identity": max_identity_cdrh3,
                           "highest CDR H3 identity name": max_identity_name_cdrh3}

        existing_overview_file = False

        for file in directory_files:
            if file.find("overview.csv") > -1:
                df = pd.read_csv(root_directory + "\\" + file)
                df.loc[len(df.index)] = written_content
                df.to_csv(root_directory + "\\" + "overview" + ".csv", header=True, index=False)
                existing_overview_file = True

        if not existing_overview_file:
            df = pd.DataFrame(columns=("schedule identity", "result number", "epoch", "highest Fv identity",
                                       "highest Fv identity name", "highest CDR identity",
                                       "highest CDR identity name", "highest CDR H3 identity", "highest CDR H3 "
                                                                                               "identity name"))

            df.loc[0] = written_content

            df.to_csv(root_directory + "\\" + "overview" + ".csv", header=True, index=False)
