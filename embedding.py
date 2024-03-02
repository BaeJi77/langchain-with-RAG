from langchain_openai import OpenAIEmbeddings
from numpy import dot
from numpy.linalg import norm

embedding_model = OpenAIEmbeddings(
    openai_api_key="OPENAI_API_KEY"
)

conversation = [
    "안녕하세요!",
    "넵! 무엇을 도와드릴까요?",
    "저의 직업은 개발자입니다.",
    "저는 개발을 잘하고 싶어요!",
]

embeddings = embedding_model.embed_documents(
    conversation
)

print(len(embeddings), len(embeddings[0]))  # 전체 임베딩 길이, 첫번째 값의 임베딩 길이

q = "대화를 나누고 있는 사람의 직업은 무엇인가요?"
a = "개발자입니다."
embedded_query_q = embedding_model.embed_query(q)
embedded_query_a = embedding_model.embed_query(a)

print(len(embedded_query_q), len(embedded_query_a))


def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))


print(q + " / " + a)
print(cos_sim(embedded_query_q, embedded_query_a))

print(q + " / " + conversation[0])
print(cos_sim(embedded_query_q, embeddings[0]))
print(q + " / " + conversation[1])
print(cos_sim(embedded_query_q, embeddings[1]))
print(q + " / " + conversation[2])
print(cos_sim(embedded_query_q, embeddings[2]))
print(q + " / " + conversation[3])
print(cos_sim(embedded_query_q, embeddings[3]))
