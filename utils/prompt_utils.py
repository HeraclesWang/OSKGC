def create_er_prompt(sent: str, example_text: str, example_output: str) -> str:
    """Create prompt for entity recognition."""
    instruction = (
        "Your task is to find all the named entities in the given text, including numbers, codes, and dates. "
        "You need to find at least two entities. Only focus on named entities and never extract adjectives or numerical units. "
        "Please strictly follow the format of the following example and only provide the named entities you find, without any additional words.\n"
    )
    example = f"Example text: {example_text}\nOutput: {example_output}\n"
    return f"{instruction}{example}Text: {sent}\nOutput:"

def create_et_prompt(text: str, entities_str: str, candidate_types: str, 
                    example_text: str, example_triples: str, example_output: str) -> str:
    """Create prompt for entity typing."""
    return f"""Please select the appropriate entity type for each given entity from the candidate entity types based on the text and your knowledge.
        Text: {text}
        Entities: {entities_str}
        Candidate entity types: {candidate_types}
        Each square bracket in the candidate entity types contains a candidate entity type. If a candidate entity type is followed by a colon, then each entity type enclosed in square brackets within the subsequent curly brace is a sub-type of that entity type.`
        Example:
        Text: {example_text}
        Entities: {example_triples}
        Output: {example_output}
        Please select only one most appropriate entity type from one of square brackets in the candidate entity types for each entity, strictly following the format shown in the example, without showing any intermediate entity types and without any additional explanation.\nOutput:"""

def create_rs_prompt(text: str, entities: str, relations: str,
                   example_text: str, example_entities: str, example_output: str,
                   example_candidate_relations: str) -> str:
    """Create prompt for relation selection."""
    return (
        f"Please select relations from the candidates to connect given entities in the text, if they exist.\n"
        f"Example:\n"
        f"Text: {example_text}\n"
        f"Given entities: {example_entities}\n"
        f"Candidate relations: {example_candidate_relations}\n"
        f"Output: {example_output}\n\n"
        f"Please strictly follow the output format in the example without any additional words.\n\n"
        f"Text: {text}\n"
        f"Given entities: {entities}\n"
        f"Candidate relations: {relations}\n\n"
        f"Output:"
    )

def create_joint_prompt(input_text: str, example_text: str, example_output: str, ontology_schema: str) -> str:
    """Create prompt for joint extraction."""
    instruction = (
        "Your task is to construct a knowledge graph from the input text based on the given ontology schema. "
        "The goal is to extract triples based on the given ontology schema's entity types, relations, and hierarchy, and provide the most accurate corresponding ontology schema possible.\n"
    )
    example = (
        f"Ontology schema:\n{ontology_schema}\n"
        f"Example text: {example_text}\n"
        f"Example output: {example_output}\n"
        "The output format must strictly follow the example, with no additional text or explanations.\n"
    )
    return f"{instruction}{example}Input text: {input_text}\nOutput:"