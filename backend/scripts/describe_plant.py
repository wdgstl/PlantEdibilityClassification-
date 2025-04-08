import openai

def get_description(plant_name, OPENAI_KEY):
    client = openai.OpenAI(api_key=OPENAI_KEY)

    prompt = f"""
        Given the following plant name do the following. Tell me what its common name is, if it is edible, and describe it: "{plant_name}"
        """

    response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

    return response.choices[0].message.content
