import re

def parse_speaker_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()


    pattern = r'([^:]+)\s*:\s*<br />(.*?)<br /><br />'
    matches = re.findall(pattern, content, re.DOTALL)

    parsed_data = []
    for speaker, text in matches:
        parsed_data.append({
            'speaker': speaker.strip(),
            'text': text.strip()
        })

    return parsed_data


file_path = 'results/llm_pipe/000_1f47ddba-8a7c-4481-a6a0-b7d94f38ca25/3_0_formatted_verbatim__format_for_output.txt'
result = parse_speaker_text(file_path)


for item in result:
    print(item)