import discotoolkit as dt
import gseapy as gp
from joblib import Parallel, delayed


def get_enrichr_geneset(organism="Human"):
    avail_organisms = ["Human", "Mouse", "Yeast", "Fly", "Fish", "Worm"]
    if organism not in avail_organisms:
        raise ValueError(f"available organism are {avail_organisms}")
    return gp.get_library_name(organism=organism)


def get_topic_enrichr(beta, geneset, topics="all", organism="human", topn_genes=20):
    enrichr = {}

    if topics == "all":
        topics = beta.columns
    elif not isinstance(topics, list):
        topics = [topics]
    for topic in topics:
        if topic not in beta.columns:
            raise KeyError(f"{topic} not Found")

    for topic in topics:
        topic_genes = beta.nlargest(topn_genes, topic).index.values.tolist()
        topic_genes = [topic_gene.upper() for topic_gene in topic_genes]
        topic_enrichr = gp.enrichr(
            gene_list=topic_genes, gene_sets=geneset, organism=organism, outdir=None
        )
        if topic not in enrichr:
            enrichr[topic] = topic_enrichr.results
    return enrichr


def get_topic_ora(beta, geneset, topics="all", topn_genes=20, n_jobs=20):
    if topics == "all":
        topics = beta.columns
    elif not isinstance(topics, list):
        topics = [topics]
    for topic in topics:
        if topic not in beta.columns:
            raise KeyError(f"{topic} not Found")

    oras = {}
    topic_genes = []
    for topic in topics:
        topic_genes.append(beta.nlargest(topn_genes, topic).index.values.tolist())

    ora = Parallel(n_jobs=n_jobs)(
        delayed(gp.enrich)(topic_gene, geneset) for topic_gene in topic_genes
    )
    i = 0
    for topic in topics:
        oras[topic] = ora[i].res2d.sort_values("P-value", inplace=False)
        i = i + 1

    return oras

    # for topic in topics:
    #     try:
    #         topic_genes = beta.nlargest(topn_genes, topic).index.values.tolist()
    #         # background = set(list(set(list(beta.index)) - set(topic_genes)))
    #         topic_ora = gp.enrich(
    #             gene_list=topic_genes, gene_sets=geneset, outdir=None, verbose=True
    #         )
    #         if topic not in ora:
    #             ora[topic] = topic_ora.results.sort_values("Adjusted P-value")
    #     except:
    #         ora[topic] = None

    return ora


def get_topic_disco(beta, topics="all", reference=None, topn_genes=20, ncores=20):
    """_summary_

    Args:
        beta (_type_): Feature by topic returned by STAMP
        topics (str, optional): Which topics to run disco on . Defaults to "all".
        reference (_type_, optional): Reference to use. None sets to all reference.
          Defaults to None.
        topn_genes (int, optional): How many top genes to run analysis on.
          Defaults to 20.
        ncores (int, optional): Number of cores to use. Defaults to 20.

    Returns:
        _type_: A dictionary with the topics and top associated genesets
    """
    if topics == "all":
        topics = beta.columns.tolist()
    elif not isinstance(topics, list):
        topics = [topics]
    for topic in topics:
        if topic not in beta.columns:
            raise KeyError(f"{topic} not Found")

    def process_topic(beta, topic, reference):
        df = beta[[topic]]
        df.columns = ["fc"]
        df["gene"] = df.index
        df = df.loc[:, ["gene", "fc"]]
        df = df.sort_values("fc", ascending=False)
        df = df.iloc[:topn_genes, :]
        df = df[["gene"]]
        return dt.CELLiD_enrichment(df, ncores=1, reference=reference)

    disco = Parallel(n_jobs=ncores)(
        delayed(process_topic)(beta, topic, reference) for topic in topics
    )

    discos = {}
    for i in range(len(topics)):
        discos[topics[i]] = disco[i]

    return discos


def get_topic_gsea(
    beta,
    geneset,
    topics="all",
    geneset_size=[5, 500],
    permutations=1000,
    n_jobs=20,
):
    if topics == "all":
        topics = beta.columns
    elif not isinstance(topics, list):
        topics = [topics]
    for topic in topics:
        if topic not in beta.columns:
            raise KeyError(f"{topic} not found")

    gsea = {}
    # library = blitz.enrichr.get_library(genesets)

    def process_topic(beta, topic, geneset):
        # Cut off sparse entries
        rank = beta.loc[:, topic]
        rank = rank.sort_values(ascending=False)
        # rank = rank.reset_index()
        results = gp.prerank(
            rank,
            geneset,
            permutations=permutations,
            min_size=geneset_size[0],
            max_size=geneset_size[1],
            threads=1,
            verbose=True,
        )
        return results.res2d

    gsea = Parallel(n_jobs=n_jobs)(
        delayed(process_topic)(beta, topic, geneset) for topic in topics
    )
    # results["Name"] = geneset

    # if topic not in gsea.keys():
    #     gsea[topic] = results
    gseas = {}
    for i in range(len(topics)):
        gseas[topics[i]] = gsea[i]

    return gseas


# def get_niches(adata, topic_prop, resolution=0.5):

#     spatial_connectivities = adata.obsp["spatial_connectivities"]
#     spatial_topic_prop = spatial_connectivities @ topic_prop
#     adata.obsm["X_spatial_latent"] = spatial_topic_prop
#     # sc.pp.neighbors(adata, use_rep = "X_spatial_latent", key_added = )
#     # sc.tl.leiden(adata)

#     return spatial_topic_prop


# def get_topic_msigdb(
#     beta, collection=["hallmark"], test="gsea", genes_filter=[-0.1, 0.1]
# ):
#     beta = beta.transpose()

#     msigdb = dc.get_resource("MSigDB")
#     print("Available options for collection are:", msigdb.collection.unique())
#     msigdb = msigdb[msigdb["collection"].isin(collection)]
#     msigdb = msigdb[~msigdb.duplicated(["geneset", "genesymbol"])]
#     return dc.run_ora(
#         mat=beta, net=msigdb, source="geneset", target="genesymbol", verbose=True
#     )
