""" Create graph embedding from dbp file


Example data : 

>>> db['1808.03258']
{'id': 'http://arxiv.org/abs/1808.03258v1', 'guidislink': True, 'link':
 'http://arxiv.org/abs/1808.03258v1', 'updated': '2018-08-04T13:37:51Z',
 'updated_parsed': time.struct_time(tm_year=2018, tm_mon=8, tm_mday=4,
 tm_hour=13, tm_min=37, tm_sec=51, tm_wday=5, tm_yday=216, tm_isdst=0),
 'published': '2018-08-04T13:37:51Z', 'published_parsed':
 time.struct_time(tm_year=2018, tm_mon=8, tm_mday=4, tm_hour=13, tm_min=37,
 tm_sec=51, tm_wday=5, tm_yday=216, tm_isdst=0), 'title': 'Application of
 Bounded Total Variation Denoising in Urban Traffic\n  Analysis',
 'title_detail': {'type': 'text/plain', 'language': None, 'base': '', 'value':
 'Application of Bounded Total Variation Denoising in Urban Traffic\n
 Analysis'}, 'summary': 'While it is believed that denoising is not always
 necessary in many big data\napplications, we show in this paper that denoising
 is helpful in urban traffic\nanalysis by applying the method of bounded total
 variation denoising to the\nurban road traffic prediction and clustering
 problem. We propose two\neasy-to-implement methods to estimate the noise
 strength parameter in the\ndenoising algorithm, and apply the denoising
 algorithm to GPS-based traffic\ndata from Beijing taxi system. For the traffic
 prediction problem, we combine\nneural network and history matching method for
 roads randomly chosen from an\nurban area of Beijing. Numerical experiments
 show that the predicting accuracy\nis improved significantly by applying the
 proposed bounded total variation\ndenoising algorithm. We also test the
 algorithm on clustering problem, where a\nrecently developed clustering
 analysis method is applied to more than one\nhundred urban road segments in
 Beijing based on their velocity profiles. Better\nclustering result is
 obtained after denoising.', 'summary_detail': {'type': 'text/plain',
 'language': None, 'base': '', 'value': 'While it is believed that denoising is
 not always necessary in many big data\napplications, we show in this paper
 that denoising is helpful in urban traffic\nanalysis by applying the method of
 bounded total variation denoising to the\nurban road traffic prediction and
 clustering problem. We propose two\neasy-to-implement methods to estimate the
 noise strength parameter in the\ndenoising algorithm, and apply the denoising
 algorithm to GPS-based traffic\ndata from Beijing taxi system. For the traffic
 prediction problem, we combine\nneural network and history matching method for
 roads randomly chosen from an\nurban area of Beijing. Numerical experiments
 show that the predicting accuracy\nis improved significantly by applying the
 proposed bounded total variation\ndenoising algorithm. We also test the
 algorithm on clustering problem, where a\nrecently developed clustering
 analysis method is applied to more than one\nhundred urban road segments in
 Beijing based on their velocity profiles. Better\nclustering result is
 obtained after denoising.'}, 'authors': [{'name': 'Shanshan Tang'}, {'name':
 'Haijun Yu'}], 'author_detail': {'name': 'Haijun Yu'}, 'author': 'Haijun Yu',
 'links': [{'href': 'http://arxiv.org/abs/1808.03258v1', 'rel': 'alternate',
 'type': 'text/html'}, {'title': 'pdf', 'href':
 'http://arxiv.org/pdf/1808.03258v1', 'rel': 'related', 'type':
 'application/pdf'}], 'arxiv_primary_category': {'term': 'cs.LG', 'scheme':
 'http://arxiv.org/schemas/atom'}, 'tags': [{'term': 'cs.LG', 'scheme':
 'http://arxiv.org/schemas/atom', 'label': None}, {'term': 'math.NA', 'scheme':
 'http://arxiv.org/schemas/atom', 'label': None}, {'term': 'stat.ML', 'scheme':
 'http://arxiv.org/schemas/atom', 'label': None}], '_rawid': '1808.03258',
 '_version': 1}
"""

from operator import itemgetter
import dgl
import torch as th
from tqdm import tqdm
import dgl.function as fn



def load_data(path):
    import pickle
    with open(path, 'rb') as fhandle:
        db = pickle.load(fhandle)
    return db


def build_graph(db, directed=True):
    print("Building graph from %d papers" % len(db.keys()))
    idx2doc = list(db.keys())
    doc2idx = {key: idx for idx, key in enumerate(idx2doc)}

    tag2idx = {}
    aut2idx = {}

    g = dgl.DGLGraph()
    g.add_nodes(len(db))

    for key in tqdm(idx2doc):
        paper = db[key]
        paper_idx = doc2idx[key]
        authors = [a['name'] for a in paper['authors']]
        tags = [t['term'] for t in paper['tags']]

        for author in authors:
            if author not in aut2idx:
                g.add_nodes(1)
                g.add_edge(len(g) - 1, paper_idx)
                aut2idx[author] = len(g) - 1
                if not directed:
                    g.add_edge(paper_idx, len(g) - 1)
            else:
                g.add_edge(aut2idx[author], paper_idx)
                if not directed:
                    g.add_edge(paper_idx, aut2idx[author])

        for tag in tags:
            if tag not in tag2idx:
                g.add_nodes(1)
                g.add_edge(paper_idx, len(g) - 1)
                tag2idx[tag] = len(g) - 1
                if not directed:
                    g.add_edge(len(g) - 1, paper_idx)
            else:
                g.add_edge(paper_idx, tag2idx[tag])
                if not directed:
                    g.add_edge(tag2idx[tag], paper_idx)

    idx2tag = {idx: tag for tag, idx in tag2idx.items()}
    idx2aut = {idx: aut for aut, idx in aut2idx.items()}

    return g, idx2doc, idx2tag, idx2aut


def top(probs, subset, n=None):
    """
    Args
    ====
    probs: array of probabilities to sort
    subset: [(idx, item)]

    Ret
    ===
    Top-n of (item, prob) pairs.
    """

    zipped_probs = [(item, probs[idx]) for idx, item in subset]
    zipped_probs.sort(key=itemgetter(1), reverse=True)
    return zipped_probs[:n]

def pagerank_main(args):
    db = load_data(args.picklefile)
    g, idx2doc, idx2tag, idx2aut = build_graph(db, directed=False)
    print("%d nodes, %d edges." % (g.number_of_nodes(), g.number_of_edges()))

    N = len(g)
    DAMP = args.damp

    print("N =", N)
    print("DAMP =", DAMP)

    def pagerank_builtin(g):
        g.ndata['pv'] = g.ndata['pv'] / g.ndata['deg']
        g.update_all(message_func=fn.copy_src(src='pv', out='m'),
                     reduce_func=fn.sum(msg='m',out='m_sum'))
        g.ndata['pv'] = (1 - DAMP) / N + DAMP * g.ndata['m_sum']

    print("Running pagerank for %d steps" % args.steps)
    # Initial values
    g.ndata['pv'] = th.ones(N) / N
    g.ndata['deg'] = g.out_degrees(g.nodes()).float()
    for k in tqdm(range(args.steps)):
        pagerank_builtin(g)

    print("Sorting...")
    # Only care for paper nodes, not tags or authors
    pv = g.ndata['pv']
    top_papers = top(pv, enumerate(idx2doc), args.top)
    print("Top %d hot papers by PageRank:" % args.top)
    for k, (key, score) in enumerate(top_papers):
        suppl_data = db[key]
        print("Top-{} (pv={:.2E}): '{}' ({}, {}, {} authors, Tags: {})".format(
            k+1, score,
            ' '.join(suppl_data['title'].split()),
            suppl_data['link'],
            suppl_data['published_parsed'].tm_year,
            len(suppl_data['authors']),
            ', '.join(t['term'] for t in suppl_data['tags'])))

    print("Top %d hot authors" % args.top)
    top_authors = top(pv, idx2aut.items(), args.top)
    for k, (key, score) in enumerate(top_authors):
        print("Top-{} (pv={:.2E}): {}".format(k+1, score, key))

    print("Top %d hot tags" % args.top)
    top_tags = top(pv, idx2tag.items(), args.top)
    for k, (key, score) in enumerate(top_tags):
        print("Top-{} (pv={:.2E}): {}".format(k+1, score, key))

    print("kthxbye.")


def pagerank_add_args(parser):
    parser.add_argument('picklefile', help="Path to pickled input data")
    parser.add_argument('-k',
                        '--steps',
                        metavar='K',
                        help="Number of pagerank steps [100]",
                        default=100,
                        type=int)

    parser.add_argument('-n',
                        '--top',
                        metavar='N',
                        help="Number of top page-ranked items to return [10]",
                        default=10,
                        type=int)

    parser.add_argument('-d', '--damp', help="Damping Factor",
                        default=.85, type=float)

def gcn_main(args):
    db = load_data(args.picklefile)
    g = build_graph(db)


def gcn_add_args(parser):
    parser.add_argument('picklefile', "Path to pickled input data")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    pagerank_add_args(parser)
    args = parser.parse_args()
    pagerank_main(args)

