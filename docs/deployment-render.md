# AdviSU — Render dağıtım rehberi

Bu proje bir monorepo'dur. Backend image'ı hem `server/` kodunu hem de `data/` altındaki resmi
veriyi içerir; bu nedenle backend Docker build context'i mutlaka repository kökü olmalıdır.
Frontend ise Vite çıktısı üreten bir Render Static Site olarak dağıtılır.

## Önerilen yöntem: Blueprint

Repository kökündeki [`render.yaml`](../render.yaml) iki servisi tanımlar:

- `advisu-backend`: `server/Dockerfile` kullanan Docker Web Service
- `advisu-frontend`: Node 22 ve pnpm ile oluşturulan Static Site

Render Dashboard'da **New > Blueprint** seçip bu repository'yi ve `render.yaml` dosyasını bağlayın.
İlk kurulumda Render, `sync: false` olarak işaretlenen şu gizli değerleri ister:

- `GROQ_API_KEY`
- `MONGO_URI`
- `ADMIN_PASSWORD`
- `STUDENT_PASSWORD`

Bu değerleri hiçbir zaman `.env`, `render.yaml`, ekran görüntüsü veya Git commit'i içine koymayın.
Blueprint `advisu-v2-dev` branch'ini izler ve her commit'ten sonra otomatik deploy başlatır.

> Var olan Render servisleri Blueprint'e bağlı değilse `render.yaml` push edilmesi onların Dashboard
> ayarlarını otomatik olarak değiştirmez. Bu durumda aşağıdaki manuel ayarları bir kez uygulayın veya
> servisleri Blueprint üzerinden yönetin.

## Mevcut backend servisini manuel düzeltme

Render Backend > Settings:

| Alan | Değer |
|---|---|
| Branch | `advisu-v2-dev` |
| Language / Runtime | `Docker` |
| Root Directory | boş |
| Dockerfile Path | `server/Dockerfile` |
| Docker Build Context Directory | `.` |
| Docker Command | boş; image'ın `CMD` satırı kullanılır |
| Health Check Path | `/test` |
| Auto-Deploy | açık (`On Commit`) |

`server/` bir klasördür ve Dockerfile Path olamaz. Context'i `server/` yapmak da hatalıdır; bu durumda
Docker `COPY server/...` ve `COPY data/...` kaynaklarını göremez.

Backend Environment bölümünde:

| Key | Değer / açıklama |
|---|---|
| `PORT` | `10000` |
| `LLM_PROVIDER` | `groq` |
| `GROQ_API_KEY` | gizli Groq anahtarı |
| `GROQ_MODEL_NAME` | `llama-3.3-70b-versatile` |
| `MONGO_URI` | gizli MongoDB Atlas URI |
| `MONGO_DB_NAME` | `advisu` |
| `CATALOG_DATA_DIR` | `/app/data` |
| `DEGREE_DATA_DIR` | `/app/data` |
| `CATALOG_TERM_CODES` | `202601` |
| `CHROMA_PERSIST_DIR` | `/app/chroma_store` |
| `DEFAULT_RETRIEVAL_MODE` | `hybrid_meta` |
| `DEFAULT_PROMPT_STRATEGY` | `structured_lookup` |
| `AUTO_SEED_COURSES` | `true` |
| `AUTO_BUILD_VECTOR_INDEX` | Free için `false` |
| `ADVISU_LAB_DENSE` | Free için `false` |
| `ADVISU_RERANK` | `off` |
| `EMBEDDING_DEVICE` | `cpu` |
| `EMBEDDING_BATCH_SIZE` | Free için `8` |
| `ADMIN_USERNAME`, `STUDENT_USERNAME` | seçilen kullanıcı adları |
| `ADMIN_PASSWORD`, `STUDENT_PASSWORD` | gizli ve güçlü parolalar |

Container komutu `0.0.0.0:${PORT:-8000}` adresine bağlanır. Böylece Render'ın `PORT=10000`
değeri ile yerel Compose'un `PORT=8000` değeri aynı image ile çalışır.

### 512 MB Free instance notu

`ingest_degree_requirements.py` yaklaşık 30 bin kaydı embed eder ve model indirir. Bunu her cold
start'ta çalıştırmak Free instance belleğini aşabilir ve ephemeral dosya sistemi nedeniyle sonuç bir
sonraki instance'a kalmaz. Blueprint bu yüzden:

- başlangıç Chroma ingest'ini kapatır (`AUTO_BUILD_VECTOR_INDEX=false`),
- ikinci dense modeli kapatır (`ADVISU_LAB_DENSE=false`),
- ölçülmüş BM25F + metadata yolunu kullanılabilir bırakır.

Tam dense/Chroma davranışı gerekiyorsa daha yüksek bellekli instance ve `/app/chroma_store` için
persistent disk kullanın ya da indeksi kontrollü bir build/job aşamasında hazırlayın. Bellek hataları
Render logunda çoğunlukla `Out of memory`, `SIGKILL` veya exit code `137` olarak görünür.

## Mevcut frontend servisini manuel düzeltme

Frontend'i **Static Site** olarak oluşturun:

| Alan | Değer |
|---|---|
| Branch | `advisu-v2-dev` |
| Root Directory | `frontend` |
| Build Command | `corepack enable && corepack prepare pnpm@10.28.0 --activate && pnpm install --frozen-lockfile && pnpm build` |
| Publish Directory | `dist` |
| `NODE_VERSION` | `22.23.1` |
| `VITE_API_URL` | `https://advisu-backend.onrender.com` |

Redirects/Rewrites bölümüne React Router için şu kuralı ekleyin:

| Action | Source | Destination |
|---|---|---|
| Rewrite | `/*` | `/index.html` |

Bu kural olmadan `/schedule`, `/profile` gibi bir client-side route doğrudan yenilendiğinde Render
404 döndürür. `VITE_API_URL` build-time değişkendir; değiştirdikten sonra frontend'i yeniden deploy
etmek gerekir.

Render pnpm lockfile gördüğünde CI modunda frozen lockfile uygular. `package.json` değiştiğinde her
zaman aşağıdakini çalıştırıp **hem** manifesti hem lockfile'ı commit edin:

```powershell
cd frontend
corepack pnpm install
corepack pnpm build
git add package.json pnpm-lock.yaml
```

`--no-frozen-lockfile` kullanmak kalıcı çözüm değildir; dağıtımın tekrar üretilebilirliğini bozar.

## Deploy sonrası doğrulama

Backend deploy tamamlandıktan sonra:

```bash
curl -fsS https://advisu-backend.onrender.com/test
```

Yanıt HTTP 200 ve `Testing successfull...` içeren JSON olmalıdır. Ardından frontend'i açıp:

1. login akışını,
2. yeni bir chat isteğini,
3. sayfayı `/schedule` veya `/profile` üzerinde doğrudan yenilemeyi,
4. Browser Network panelinde API isteklerinin backend URL'sine gittiğini

kontrol edin. Docker/path ayarı değiştirildiyse ilk sefer **Clear build cache & deploy** kullanın;
normal kod commit'lerinde standart auto-deploy yeterlidir.

## Yerel eşdeğer

```powershell
Copy-Item .env.example .env
# .env içindeki gerçek provider anahtarını ekleyin
docker compose up --build
```

- Frontend: <http://localhost:5173>
- Backend health: <http://localhost:8000/test>
- API docs: <http://localhost:8000/docs>

Yerelde named volume kullanıldığı için `AUTO_BUILD_VECTOR_INDEX=true` varsayılanı uygundur. Verileri
bilerek silmek istemedikçe `docker compose down -v` çalıştırmayın.
