CREATE CONSTRAINT jurisdictionId IF NOT EXISTS on (jur:Jurisdiction) ASSERT jur.id IS UNIQUE;
LOAD CSV WITH HEADERS FROM 'file:///jurisdictions.csv' AS row
WITH row
MERGE (jurisdiction:Jurisdiction {id:toInteger(row.id)})
ON CREATE SET jurisdiction.name = row.name_long,
jurisdiction.url = row.url;

CREATE CONSTRAINT courtId IF NOT EXISTS on (cou:Court) ASSERT cou.id IS UNIQUE;
LOAD CSV WITH HEADERS FROM 'file:///courts.csv' AS row
WITH row
MERGE (court:Court {id: toInteger(row.id)})
ON CREATE SET court.name = row.name,
court.url = row.url;

CREATE CONSTRAINT reporterId IF NOT EXISTS on (rep:Reporter) ASSERT rep.id IS UNIQUE;
LOAD CSV WITH HEADERS FROM 'file:///reporters.csv' AS row
WITH row
MERGE (reporter:Reporter {id: toInteger(row.id)})
ON CREATE SET reporter.name = row.full_name,
reporter.url = row.url;

CREATE CONSTRAINT volumeId IF NOT EXISTS on (vol:Volume) ASSERT vol.id IS UNIQUE;
LOAD CSV WITH HEADERS FROM 'file:///volumes.csv' AS row
WITH row
MERGE (volume:Volume {id: toInteger(row.volume_number)})
ON CREATE SET volume.barcode = row.barcode,
volume.url = volume.url;

CREATE CONSTRAINT caselawId IF NOT EXISTS on (cas:Caselaw) ASSERT cas.id IS UNIQUE;
:auto USING PERIODIC COMMIT 1000
LOAD CSV WITH HEADERS FROM 'file:///cases.csv' AS row
WITH row
MATCH (volume:Volume {id: toInteger(row.volume_id)})
MATCH (court:Court {id: toInteger(row.court_id)})
MATCH (reporter:Reporter {id: toInteger(row.reporter_id)})
MATCH (jurisdiction:Jurisdiction {id: toInteger(row.jurisdiction_id)})
MERGE (caselaw:Caselaw {id: toInteger(row.id)})
MERGE (caselaw) -[:REPORTED_BY]->(reporter)
MERGE (caselaw) -[:HEARD_BY]->(court)
MERGE (caselaw) -[:IN_VOLUME]->(volume)
MERGE (caselaw) -[:UNDER_JURISIDICTION]->(jurisdiction)
on CREATE SET caselaw.decision_date = datetime(row.decision_date),
caselaw.name = row.name,
caselaw.url = row.url;

:auto USING PERIODIC COMMIT 1000
LOAD CSV WITH HEADERS FROM 'file:///citations.csv' AS row
with row
MATCH (case1: Caselaw {id: toInteger(row.src)})
MATCH (case2: Caselaw {id: toInteger(row.dst)})
MERGE (case1) -[:CITED]-> (case2);





//Queries
MATCH g=(j1:Jurisdiction)<-[:UNDER_JURISIDICTION]-(c1:Caselaw)<-[:CITED]-(c2),
(c1)-[:HEARD_BY]->(court:Court)
RETURN j1.name as Jur, court.name as court, c1.name as caselaw, count(DISTINCT c2) as citations
ORDER BY citations DESC
LIMIT 25;



MATCH g=(j1:Jurisdiction)<-[:UNDER_JURISIDICTION]-(c1:Caselaw)<-[:CITED]-(c2)
with j1, c1, count(DISTINCT c2) as citations
ORDER BY citations desc
WITH j1.name as Jur, collect({caselaw: c1.name, citations: citations}) as top_k
RETURN Jur, top_k[0..5]
ORDER BY Jur DESC
LIMIT 5;

//ARTICLE RANK algorithm
CALL gds.graph.create('citation-network', 'Caselaw', 'CITED');
CALL gds.alpha.articleRank.write('citation-network',
  {writeProperty:'articlerank_citation'});

// Article rank score
MATCH (a:Caselaw)
RETURN a.name as title, a.articlerank_citation as articlerank
ORDER BY articlerank
DESC LIMIT 10;

//Degree centrality algorithm
CALL gds.alpha.degree.stream({
  nodeProjection: 'Caselaw',
  relationshipProjection: {
    Cited: {
      type: 'CITED',
      orientation: 'REVERSE'
    }
  }
})
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS name, score AS citations
ORDER BY citations DESC