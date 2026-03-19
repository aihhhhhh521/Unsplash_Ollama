from pathlib import Path

# ========= 基本路径 =========
# 改成你自己的 Unsplash 数据集目录
DATA_ROOT = Path(r"D:/PyProjects/Dataset/unsplash-research-dataset-full-latest").resolve()
WORK_DIR = DATA_ROOT / "work"
WORK_DIR.mkdir(parents=True, exist_ok=True)

# ========= Ollama 配置 =========
OLLAMA_BASE_URL = "http://localhost:11434/api"
OLLAMA_MODEL = "gemma3"   # 可改成你本机已经 pull 好的模型
OLLAMA_KEEP_ALIVE = "30m"
OLLAMA_READ_BATCH_SIZE = 10000
OLLAMA_RECORDS_PER_REQUEST = 16
OLLAMA_MAX_IN_FLIGHT_BATCHES = 16
OLLAMA_WRITE_BUFFER_SIZE = 1000

# 速度优先时建议关掉
OLLAMA_RETURN_REASON = False

# 单轮分类更推荐 generate
OLLAMA_USE_GENERATE_API = True

# 先保守一点，别盲目拉太高
MAX_WORKERS = 12
REQUEST_TIMEOUT = 180
MAX_RETRIES = 3

# ========= 数据处理配置 =========
PARQUET_BATCH_SIZE = 20000
TEXT_MAX_CHARS = 1800

# ========= 规则阈值 =========
RULE_DIRECT_MIN_SCORE = 4
RULE_DIRECT_MIN_MARGIN = 2
REVIEW_CONFIDENCE_THRESHOLD = 0.70

# ========= 明确需要过滤的 art 关键词 =========
ART_KEYWORDS = [
    "art",
    "artwork",
    "painting",
    "illustration",
    "drawing",
    "sculpture",
    "mural",
    "graffiti",
    "sketch",
    "cartoon",
    "anime",
    "poster",
    "collage",
    "calligraphy",
    "watercolor",
    "oil painting",
    "installation",
]

# ========= 五类标签 =========
LABELS = ["城市、建筑", "室内", "自然", "静物", "人像"]

# ========= 规则词典 =========
# 注意：这是“预分类词典”，不是最终真理。作用是尽量减少送去 Ollama 的样本量。
CATEGORY_VOCAB = {
    "人像": {
        "strong": {
            "portrait", "face", "selfie", "model", "bride", "groom", "headshot",
            "close up face", "close-up face", "profile portrait"
        },
        "weak": {
            "person", "people", "human", "woman", "man", "girl", "boy", "child",
            "baby", "couple", "family", "male", "female"
        },
    },
    "室内": {
        "strong": {
            "interior", "indoor", "bedroom", "living room", "kitchen", "bathroom",
            "office", "library", "hotel room", "apartment", "hallway", "classroom"
        },
        "weak": {
            "room", "cafe", "restaurant", "hotel", "studio", "desk", "indoors"
        },
    },
    "城市、建筑": {
        "strong": {
            "architecture", "skyline", "skyscraper", "building exterior", "street scene",
            "city street", "urban landscape", "bridge", "tower", "cathedral", "temple"
        },
        "weak": {
            "city", "urban", "street", "building", "downtown", "facade", "road",
            "campus", "church", "palace", "station", "landmark"
        },
    },
    "自然": {
        "strong": {
            "landscape", "mountain", "waterfall", "forest", "beach", "ocean",
            "sea", "river", "lake", "sunset", "sunrise", "wildlife"
        },
        "weak": {
            "nature", "tree", "flower", "plant", "animal", "bird", "dog", "cat",
            "horse", "sky", "snow", "desert", "valley", "grass"
        },
    },
    "静物": {
        "strong": {
            "still life", "product photo", "food photography", "tabletop"
        },
        "weak": {
            "food", "drink", "coffee", "tea", "fruit", "camera", "phone", "laptop",
            "watch", "bottle", "cup", "plate", "table", "chair", "product", "object",
            "book", "car", "motorcycle"
        },
    },
}

# ========= 规则打分权重 =========
RULE_WEIGHTS = {
    "keyword_strong": 5,
    "keyword_weak": 3,
    "text_strong": 3,
    "text_weak": 1,
}

# ========= 文件名 =========
MANIFEST_FILE = WORK_DIR / "manifest.parquet"
PHOTOS_NO_ART_FILE = WORK_DIR / "photos_no_art.parquet"
REMOVED_ART_FILE = WORK_DIR / "removed_art.parquet"
PRECLASSIFIED_FILE = WORK_DIR / "preclassified.parquet"
NEED_LLM_FILE = WORK_DIR / "need_llm.parquet"
OLLAMA_RESULTS_JSONL = WORK_DIR / "ollama_results.jsonl"
OLLAMA_RESULTS_FILE = WORK_DIR / "ollama_results.parquet"
CLASSIFIED_FILE = WORK_DIR / "classified.parquet"
NEED_REVIEW_FILE = WORK_DIR / "need_review.parquet"
STATS_FILE = WORK_DIR / "category_stats.csv"
