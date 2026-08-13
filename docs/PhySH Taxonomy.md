PhySH blends standard **[SKOS](https://www.w3.org/TR/skos-reference/) (Simple Knowledge Organization System)** taxonomy (broader/narrower) with custom **PhySH-specific structural predicates** like `inDiscipline`, `inFacet`, and `hasConcept` for a more connected and complete hierarchy for tag consolidation.

From the PhySH REST API [official documentation](https://app.swaggerhub.com/apis/apsphysics/physh/1.0.0) which describes the custom hierarchical structure
- Name: "facet"
	Description: "Facets are broad groupings of concepts according to the general role they serve"
- Name: "discipline"
	Description: "Disciplines are specialties within physics used to narrow the list of concepts"
- Name: "concept"
	Description: "Concepts are the fundamental building blocks used for classification"

					[ Discipline / Facet ]
              (e.g., Condensed Matter Physics)
                           │
            ┌───────────────┴───────────────┐
	   physh_rdf:hasConcept             physh_rdf:inDiscipline
            │                               │
            ▼                               │
      [ Concept A ] ◄───────────────────────┘
            │
      skos:narrower / skos:broader
            │
            ▼
      [Concept B]

# Parsing the rdf structure

Unique predicates appearing in physh.rdf revealing the rich tree structure between concepts. They can be classified into 3 categories: 
1. SKOS which primarily defines the relationships between concepts, 
	 1. rdf:type - each entry can be either SKOS.Concept, SKOS.ConceptScheme, PhySH.Concept, PhySH.Discipline or PhySH.Facet 
	 2. skos:related
	 3. skos:narrower
	 4. skos:broader
	 5. skos:prefLabel
	 6. skos:altLabel
	 7. skos:scopeNote
	 8. skos:hiddenLabel
	 9. skos:hasTopConcept
	 10. skos:example
	 11. skos:topConceptOf
 2. PHYSH_RDF [custom ontology](https://github.com/physh-org/PhySH/blob/master/README.md) pertaining to APS publications
	 1. physh_rdf:inDiscipline
	 2. physh_rdf:prefLabel
	 3. physh_rdf:hasConcept
	 4. physh_rdf:contains
	 5. physh_rdf:inFacet
	 6. physh_rdf:usesFacet
	 7. physh_rdf:usedByDiscipline
	 8. physh_rdf:deprecated
	 9. physh_rdf:excludeFromIndexing
 3. DCTERMS (Dublin core terms vocabulary) for  concept schemes/disciplines metadata
	 1. dcterms:publisher
	 2. dcterms:description
	 3. dcterms:title
	 4. dcterms:subject

When creating the NetworkX graph, the relationship type is stored on the edge. There are 3 kinds of rel_type:
1. SKOS Concept parent
2. PHYSH Facet parent
3. PHYSH Discipline parent

If a particular concept appears less frequently, then its parent will be decided in the following hierarchy:  

```
SKOS Concept parent → PHYSH Discipline parent → PHYSH Facet parent.
```
For a child with multiple SKOS parents, it is replaced by all of the parents.