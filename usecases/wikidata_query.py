from SPARQLWrapper import SPARQLWrapper, JSON

def run_wikidata_query(sparql_query):
    # Connect to the Wikidata SPARQL endpoint
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    
    # Execute the query and fetch the results
    results = sparql.query().convert()
    
    # Parse and return the query results
    parsed_results = []
    for result in results["results"]["bindings"]:
        item = result["item"]["value"]
        label = result["itemLabel"]["value"]
        parsed_results.append((item, label))
    
    return parsed_results