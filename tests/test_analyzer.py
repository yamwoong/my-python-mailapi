from app.services.analyzer import analyze_text

# Test with a positive Korean sentence
print(analyze_text("오늘 날씨가 정말 좋네요. 기분이 좋아요."))

# Test with a positive English sentence
print(analyze_text("I am so happy to meet you!"))

# Test with too short input (should return an error)
print(analyze_text("짧다"))

# Test with an unsupported language (Japanese)
print(analyze_text("これは日本語のテストです。"))
