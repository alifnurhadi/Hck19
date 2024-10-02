def format_response(response_dict):
    query = response_dict['query']
    text = response_dict['text']
    # context = response_dict.get('context', '')  # Use .get() in case 'context' is not always present
    
    formatted_string = f"""
    Pertanyaan pengguna: {query}

    Jawaban: {text}

    """
    
    return formatted_string.strip()

# def format_response(response_dict):
#     query = response_dict['query']
#     text = response_dict['text']
#     context = response_dict.get('context', '')  # Use .get() in case 'context' is not always present
    
#     formatted_string = f"""
#     Pertanyaan pengguna: {query}

#     Jawaban: {text}

#     Informasi tambahan:
#     {context}
#     """
    
#     return formatted_string.strip()