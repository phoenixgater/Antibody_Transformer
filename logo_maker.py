# Generating a sequence logo to visualise the confidence with which predicted amino acid residues are chosen
import logomaker as lm
import pandas as pd
import matplotlib.pyplot as plt
import Write_Results


def create_probability_logo(prob_dict, vocabulary, write_directory, epoch, model_type=False, result_id = 1, sep_id=False, eod_id=False):
    if model_type == "NMT":
        vocabulary_2 = vocabulary.copy()
        vocabulary_2["J"] = vocabulary_2.pop("<OTHER>")
        vocabulary_2.pop("<SOS>")
        vocabulary_2.pop("<EOS>")
        vocabulary_2.pop("<PAD>")

    try:

        if not model_type:
            vocabulary_upper = []
            for token in vocabulary:
                vocabulary_upper.append(token.upper())
            df_heavy = pd.DataFrame(columns=vocabulary_upper[2:], dtype=int)  # "2:" to exclude UNK and pad tokens
            df_light = pd.DataFrame(columns=vocabulary_upper[2:], dtype=int)

            heavy_chain_dict_keys = (list(prob_dict.keys())[:sep_id])
            light_chain_dict_keys = (list(prob_dict.keys())[sep_id + 1:eod_id])

        else:
            vocabulary_upper = []
            for token in vocabulary_2.keys():
                vocabulary_upper.append(token.upper())
            df_heavy = pd.DataFrame(columns=vocabulary_upper, dtype=int)
            df_light = pd.DataFrame(columns=vocabulary_upper, dtype=int)

            heavy_chain_dict_keys = (list(prob_dict.keys())[:sep_id + 1])
            light_chain_dict_keys = (list(prob_dict.keys())[sep_id + 1:eod_id])

        for key in heavy_chain_dict_keys:
            for column in df_heavy.columns:
                if str(prob_dict[key][1]).lower() == column.lower():
                    df_heavy.loc[int(key), column] = prob_dict[key][0]
                else:
                    df_heavy.loc[int(key), column] = 0
        df_heavy.index = range(1, sep_id + 1)  # Sequence length 142

        for key in light_chain_dict_keys:
            for column in df_light.columns:
                if str(prob_dict[key][1]).lower() == column.lower():
                    df_light.loc[int(key), column] = prob_dict[key][0]
                else:
                    df_light.loc[int(key), column] = 0
        df_light.index = range(1, eod_id - sep_id)  # Sequence length 116

        # Creating dataframes with no J tokens - if a non-flanking "J" token is detected, J tokens are instead included
        internal_j = False
        '''df_heavy_2 = pd.DataFrame.copy(df_heavy)
        for row in df_heavy_2.itertuples():
            if row.J > 0.0:
                if row[0] < 142:
                    if df_heavy_2.loc[row[0] + 1, "J"] == 0.0:
                        internal_j = True
                        break
                    else:
                        df_heavy_2.drop(row[0], inplace=True)
                else:
                    df_heavy_2.drop(row[0], inplace=True)
    
        df_heavy_2.reset_index(drop=True, inplace=True)
    
        df_light_2 = pd.DataFrame.copy(df_light)
        for row in df_light_2.itertuples():
            if row.J > 0.0:
                if row[0] < 116:
                    if df_light_2.loc[row[0] + 1, "J"] == 0.0:
                        internal_j = True
                        break
                    else:
                        df_light_2.drop(row[0], inplace=True)
                else:
                    df_light_2.drop(row[0], inplace=True)
    
        if internal_j:
            pass
        else:
            df_heavy = df_heavy_2
            df_light = df_light_2
    '''
        logo_heavy = lm.Logo(df_heavy, font_name='Arial Rounded MT Bold', color_scheme="skylign_protein", figsize=(45, 5))
        logo_heavy.ax.set_xlabel('Position', fontsize=14)
        logo_heavy.ax.set_ylabel("Probability", labelpad=1, fontsize=14)
        Write_Results.write_image(logo_heavy.fig, "logo_heavy" + "_" + str(result_id), write_directory, epoch)

        plt.figure()
        logo_light = lm.Logo(df_light, font_name='Arial Rounded MT Bold', color_scheme="skylign_protein", figsize=(45, 5))
        logo_light.ax.set_xlabel('Position', fontsize=14)
        logo_light.ax.set_ylabel("Probability", labelpad=1, fontsize=14)
        Write_Results.write_image(logo_light.fig, "logo_light" + "_" + str(result_id), write_directory, epoch)
        plt.figure()
    except IndexError:
        pass




