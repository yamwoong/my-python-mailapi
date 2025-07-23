from app.services.analyzer import analyze_text, summarize_text

# Test with a longer positive Korean sentence
text_ko = "오늘 날씨가 정말 좋네요. 친구들과 공원에 가서 산책도 하고 이야기도 많이 나눴어요. 기분이 아주 좋아졌어요."

# Test with a longer positive English sentence
text_en = "I am so happy to meet you! Today was a wonderful day, I went for a walk in the park and talked a lot with my friends."

# Sentiment analysis tests
print("== Korean Sentiment ==")
print(analyze_text(text_ko))

print("== English Sentiment ==")
print(analyze_text(text_en))

# Summarization tests
print("== Korean Summary ==")
print(summarize_text(text_ko))

print("== English Summary ==")
print(summarize_text(text_en))

# Test with too short input for summarization (should return an error)
print("== Too Short for Summary ==")
print(summarize_text("짧다"))

# Test with an unsupported language (Japanese)
print("== Unsupported Language (Japanese) ==")
print(analyze_text("これは日本語のテストです。"))
