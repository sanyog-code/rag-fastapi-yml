from app.rag_pipeline import create_rag_pipeline

print("🚀 CI/CD FAISS build started")

# Force FAISS creation even in CI
create_rag_pipeline(build_index=True)

print("✅ FAISS index successfully created")