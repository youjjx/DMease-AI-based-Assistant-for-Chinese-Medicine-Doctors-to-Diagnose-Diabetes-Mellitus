"""Build an identifier-resolved herb/compound/structure chain; resumable API cache."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time

import requests
from rdkit import Chem

from dmease.affinity_research import standardized_structure
from dmease.evidence_policy import EvidenceGraph

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/v5'
RAW = ROOT/'data/raw/public/chemistry_v5'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8', newline='\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url, params=None, method='GET', attempts=3):
    key = hashlib.sha256(json.dumps([method, url, params], sort_keys=True).encode()).hexdigest()
    path = RAW/(key+'.json'); meta_path = RAW/(key+'.meta.json')
    if path.exists() and meta_path.exists():
        meta = read(meta_path)
        if sha(path) != meta['sha256']:
            raise ValueError('Cached source checksum mismatch: '+str(path))
        return read(path), meta
    for attempt in range(attempts):
        try:
            time.sleep(.3)  # Sequential PubChem requests stay below 5 requests/s.
            response = requests.request(method, url, params=params if method=='GET' else None,
                                        data=params if method=='POST' else None, timeout=45)
            response.raise_for_status(); payload = response.json()
            save(path, payload)
            meta = {'url': url, 'params': params, 'method': method, 'final_url': response.url,
                    'retrieved_at': datetime.now(timezone.utc).isoformat(), 'sha256': sha(path),
                    'bytes': path.stat().st_size, 'cache_file': path.name}
            save(meta_path, meta)
            return payload, meta
        except (requests.RequestException, ValueError):
            if attempt+1 == attempts:
                raise
            time.sleep(2**attempt)


def crosswalk():
    graph = EvidenceGraph(ROOT/'data/processed/evidence_bundle.json')
    db = ROOT/'data/processed/public_evidence.sqlite'
    connection = sqlite3.connect(db.as_uri()+'?mode=ro', uri=True)
    by_tcmsp = defaultdict(list)
    for (text,) in connection.execute("SELECT payload FROM entities WHERE source='SymMap' AND version='2.0' AND kind='SMIT'"):
        row = json.loads(text)
        for identifier in re.findall(r'MOL\d{6}', str(row.get('TCMSP_id') or '')):
            by_tcmsp[identifier].append(row)
    herbs = {h['name']: h for h in graph.herbs}; links = []; audit = Counter(); unresolved = []
    for subject, obj, provenance in connection.execute("SELECT subject,object,provenance FROM edges WHERE source='TCMSP' AND relation='contains' ORDER BY subject,object"):
        name = subject.removeprefix('TCMSP:herb:')
        if name not in herbs:
            continue
        source = json.loads(provenance); molecule = source['row']['MOL_ID']
        rows = [r for r in by_tcmsp[molecule] if r.get('Suppress') == 0]
        # A many-to-one explicit mapping is fine; multiple CIDs remain unresolved.
        cids = set()
        malformed = False
        for row in rows:
            value = str(row.get('PubChem_CID') or '').strip()
            if value and not re.fullmatch(r'\d+(?:\|\d+)*', value):
                malformed = True
            cids.update(int(x) for x in value.split('|') if x.isdigit() and int(x)>0)
        reason = ('no_active_crossref' if not rows else 'invalid_cid' if malformed else
                  'missing_cid' if not cids else 'ambiguous_cid' if len(cids)!=1 else None)
        audit[reason or 'resolved_cid'] += 1
        entry = {'herb_id': herbs[name]['id'], 'herb': name, 'tcmsp_id': molecule,
                 'compound_name': source['row']['molecule_name'], 'herb_source_url': source['url'],
                 'tcmsp_molecular_weight': source['row'].get('mw'),
                 'symmap_formulas': sorted({str(r['Molecule_formula']).strip() for r in rows if r.get('Molecule_formula')}),
                 'symmap_ids': [f"SMIT{int(r['Mol_id']):05d}" for r in rows]}
        if reason:
            unresolved.append({**entry, 'reason': reason, 'candidate_cids': sorted(cids)})
        else:
            links.append({**entry, 'cid': next(iter(cids))})
    connection.close()
    save(OUT/'identifier_links.json', {'evidence_fingerprint': graph.fingerprint, 'database_sha256': sha(db),
         'crossref_source': 'SymMap v2.0 official SMIT download: TCMSP_id and PubChem_CID; suppressed rows excluded',
         'audit': dict(audit), 'links': links, 'unresolved': unresolved,
         'herbs_without_tcmsp_contains': [h['name'] for h in graph.herbs if h['name'] not in {r['herb'] for r in links+unresolved}]})
    print('Crosswalk:', dict(audit), 'unique CIDs', len({r['cid'] for r in links}), flush=True)


def structures():
    data = read(OUT/'identifier_links.json'); ids = sorted({r['cid'] for r in data['links']})
    records, failures, manifest = [], [], []
    def batch(cids):
        url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/'+','.join(map(str,cids))+'/property/SMILES,InChI,InChIKey,MolecularFormula,MolecularWeight/JSON'
        try:
            result, meta = fetch(url); manifest.append(meta)
        except (requests.RequestException, ValueError) as exc:
            if len(cids)>1:
                middle=len(cids)//2; batch(cids[:middle]); batch(cids[middle:]); return
            failures.append({'cid':cids[0], 'reason':type(exc).__name__}); return
        returned = set()
        for prop in result.get('PropertyTable',{}).get('Properties',[]):
            cid = prop['CID']; returned.add(cid)
            try:
                mol = Chem.MolFromSmiles(prop['SMILES'])
                if mol is None or Chem.MolToInchiKey(mol)!=prop['InChIKey']:
                    raise ValueError('SMILES and InChIKey disagree')
                canonical, scaffold = standardized_structure(prop['SMILES'])
                parent_key = Chem.MolToInchiKey(Chem.MolFromSmiles(canonical))
                records.append({'cid':cid, **prop, 'parent_smiles':canonical, 'parent_inchikey':parent_key,
                                'scaffold':scaffold, 'source_sha256':meta['sha256'],
                                'source_url':f'https://pubchem.ncbi.nlm.nih.gov/compound/{cid}'})
            except (ValueError, RuntimeError, KeyError) as exc:
                failures.append({'cid':cid,'reason':str(exc)})
        for cid in set(cids)-returned:
            failures.append({'cid':cid,'reason':'CID not returned (no silent replacement)'})
    for start in range(0,len(ids),30):
        batch(ids[start:start+30]); print('Structures',min(start+30,len(ids)), '/',len(ids), 'valid',len(records),flush=True)
        save(OUT/'structures.json', {'identifier_links_sha256':sha(OUT/'identifier_links.json'),
             'records':records,'failures':failures,'requests':manifest})


def formula_elements(formula):
    if not re.fullmatch(r'(?:[A-Z][a-z]?\d*)+',formula or ''):return None
    counts=Counter()
    for element,count in re.findall(r'([A-Z][a-z]?)(\d*)',formula):counts[element]+=int(count or 1)
    return dict(counts)


def identity():
    links=read(OUT/'identifier_links.json')['links']
    structures={s['cid']:s for s in read(OUT/'structures.json')['records']}
    accepted=[];rejected=[]
    for link in links:
        if link['cid'] not in structures:
            rejected.append({**link,'reasons':['structure_unavailable']});continue
        structure=structures[link['cid']];reasons=[]
        mass=link.get('tcmsp_molecular_weight');pubchem_mass=float(structure['MolecularWeight'])
        if mass and abs(float(mass)-pubchem_mass)>max(1.,.005*pubchem_mass):reasons.append('molecular_weight_conflict')
        expected=formula_elements(structure['MolecularFormula'])
        for formula in link['symmap_formulas']:
            observed=formula_elements(formula)
            if expected and observed and expected!=observed:reasons.append('molecular_formula_conflict');break
        if reasons:rejected.append({**link,'reasons':reasons,'pubchem_formula':structure['MolecularFormula'],'pubchem_mw':pubchem_mass})
        else:accepted.append(link)
    save(OUT/'identity_audit.json',{'identifier_links_sha256':sha(OUT/'identifier_links.json'),
         'structures_sha256':sha(OUT/'structures.json'),'accepted':accepted,'rejected':rejected,
         'criteria':'Explicit unique active crossref plus compatible molecular weight within max(1 Da,0.5%) and matching elemental counts when parseable; exclude every conflicting path. Missing formula is not confirmation. Original TCMSP stereostructures were not independently retrieved.',
         'counts':{'accepted_links':len(accepted),'rejected_links':len(rejected),'accepted_cids':len({r['cid'] for r in accepted}),
                   'reasons':dict(Counter(reason for r in rejected for reason in r['reasons']))}})
    print('Identity audit',read(OUT/'identity_audit.json')['counts'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['crosswalk','structures','identity']);args=parser.parse_args()
    globals()[args.stage]()
