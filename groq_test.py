from groq import Groq

client = Groq()
completion = client.chat.completions.create(
    model="llama3-8b-8192",
    messages=[
        {
            "role": "system",
            "content": "You are helpful assistant who knows about wars."
        },
        {
            "role": "user",
            "content": "When is next world war coming?"
        },
        {
            "role": "assistant",
            "content": "It is difficult to predict with certainty when or if a next world war will come. However, there are several factors that could increase the risk of a global conflict:\n\n1. Geopolitical tensions: Rising tensions between major world powers, such as the United States, China, and Russia, could escalate into a global conflict.\n2. Regional conflicts: Ongoing conflicts in several regions, such as the Middle East, Asia, and Africa, could spread and become more widespread.\n3. Nuclear proliferation: The spread of nuclear weapons could lead to an increased risk of global conflict.\n4. Cyber warfare: The increasing use of cyber attacks could lead to a shift in the nature of warfare and potentially lead to a global conflict.\n5. Economic inequality: Growing economic disparities between countries and within countries could lead to social unrest and potentially escalate into a global conflict.\n\nIt's important to note that while these factors could contribute to a higher risk of a global conflict, they are not necessarily a guarantee that a war will occur."
        }
    ],
    temperature=1,
    max_tokens=1024,
    top_p=.9,
    stream=True,
    stop=None,
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
