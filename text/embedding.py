from model2vec import StaticModel

model = StaticModel.from_pretrained("minishlab/potion-base-4M")
embeddings = model.encode_as_sequence(["Example sentence"])

print(embeddings)