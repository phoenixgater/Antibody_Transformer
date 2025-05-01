# Querying the SAbDab database with a generated Fv design
import requests as req


def query_sabdab(hchain_seq, lchain_seq):
    url = "http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/sequencesearch/"
    heavy_chain_query = hchain_seq
    light_chain_query = lchain_seq

    # Query whole Fv sequence
    query = {'hchain': heavy_chain_query, 'lchain': light_chain_query, 'nostructures': 10, 'minseqid': 70,
             'region': 'Full variable region'}
    whole_fv_response = req.get(url, params=query)
    whole_fv_response_url = whole_fv_response.url
    whole_fv_matches = (req.get(whole_fv_response_url))
    matched_fv_results = {}

    # Retrieve matched PDB IDs and their shared identity with the Fv query
    for line in whole_fv_matches.text.splitlines():
        prefix = 'href="/webapps/newsabdab/sabdab/structureviewer/?pdb='
        pdb_id_pos = line.find(prefix)
        if pdb_id_pos > -1:
            pdb_id = line[pdb_id_pos + len(prefix): pdb_id_pos + len(prefix) + 4]
            matched_fv_results[pdb_id] = []
            matched_fv_results[pdb_id].append(float(line[-11: -5].replace(">", "")))

    # Retrieve name of matched PDB file
    for pdb_id in matched_fv_results.keys():
        sab_dab_record = req.get('http://opig.stats.ox.ac.uk/' + 'webapps/newsabdab/sabdab/structureviewer/?pdb=' +
                                 pdb_id)
        for line in sab_dab_record.text.splitlines():
            prefix = '<p class="lead">'
            name_pos = line.find(prefix)
            if name_pos > -1:
                if line.find("This PDB has ") > -1:
                    pass
                elif line.find("Additional links and files") > -1:
                    pass
                else:
                    matched_fv_results[pdb_id].append(line[len(prefix) + name_pos: -4])

    # Query CDRs only
    query = {'hchain': heavy_chain_query, 'lchain': light_chain_query, 'nostructures': 10, 'minseqid': 70,
             'region': 'CDRs'}
    cdr_response = req.get(url, params=query)
    cdr_response_url = cdr_response.url
    cdr_matches = (req.get(cdr_response_url))
    matched_cdr_results = {}

    # Retrieve matched PDB IDs and their shared identity with the CDR query
    prefix = 'href="/webapps/newsabdab/sabdab/structureviewer/?pdb='
    for line in cdr_matches.text.splitlines():
        prefix = 'href="/webapps/newsabdab/sabdab/structureviewer/?pdb='
        pdb_id_pos = line.find(prefix)
        if pdb_id_pos > -1:
            pdb_id = line[pdb_id_pos + len(prefix): pdb_id_pos + len(prefix) + 4]
            matched_cdr_results[pdb_id] = []
            matched_cdr_results[pdb_id].append(float(line[-11: -5].replace(">", "")))

    # Retrieve name of matched PDB file
    for pdb_id in matched_cdr_results.keys():
        sab_dab_record = req.get('http://opig.stats.ox.ac.uk/' + 'webapps/newsabdab/sabdab/structureviewer/?pdb=' +
                                 pdb_id)
        for line in sab_dab_record.text.splitlines():
            prefix = '<p class="lead">'
            name_pos = line.find(prefix)
            if name_pos > -1:
                if line.find("This PDB has ") > -1:
                    pass
                elif line.find("Additional links and files") > -1:
                    pass
                else:
                    matched_cdr_results[pdb_id].append(line[len(prefix) + name_pos: -4])

    # CDR H3 only
    query = {'hchain': heavy_chain_query, 'lchain': light_chain_query, 'nostructures': 10, 'minseqid': 70,
             'region': 'CDRH3'}
    cdrh3_response = req.get(url, params=query)
    cdrh3_response_url = cdrh3_response.url
    cdrh3_matches = (req.get(cdrh3_response_url))
    matched_cdrh3_results = {}

    # Retrieve matched PDB IDs and their shared identity with the CDR query
    prefix = 'href="/webapps/newsabdab/sabdab/structureviewer/?pdb='
    for line in cdrh3_matches.text.splitlines():
        prefix = 'href="/webapps/newsabdab/sabdab/structureviewer/?pdb='
        pdb_id_pos = line.find(prefix)
        if pdb_id_pos > -1:
            pdb_id = line[pdb_id_pos + len(prefix): pdb_id_pos + len(prefix) + 4]
            matched_cdrh3_results[pdb_id] = []
            matched_cdrh3_results[pdb_id].append(float(line[-11: -5].replace(">", "")))

    # Retrieve name of matched PDB file
    for pdb_id in matched_cdrh3_results.keys():
        sab_dab_record = req.get('http://opig.stats.ox.ac.uk/' + 'webapps/newsabdab/sabdab/structureviewer/?pdb=' +
                                 pdb_id)
        for line in sab_dab_record.text.splitlines():
            prefix = '<p class="lead">'
            name_pos = line.find(prefix)
            if name_pos > -1:
                if line.find("This PDB has ") > -1:
                    pass
                elif line.find("Additional links and files") > -1:
                    pass
                else:
                    matched_cdrh3_results[pdb_id].append(line[len(prefix) + name_pos: -4])

    if len(matched_fv_results) == 0:
        matched_fv_results = {"None": [0, "None"]}
    if len(matched_cdr_results) == 0:
        matched_cdr_results = {"None": [0, "None"]}
    if len(matched_cdrh3_results) == 0:
        matched_cdrh3_results = {"None": [0, "None"]}

    return matched_fv_results, matched_cdr_results, matched_cdrh3_results








