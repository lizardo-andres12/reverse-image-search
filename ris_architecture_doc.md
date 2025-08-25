# Reverse Image Search MVP - Architecture Design Document

## 1. Overview

**Project**: AI-powered Reverse Image Search Application  
**Goal**: Allow users to upload images and find visually similar images with keyword extraction and source attribution  
**MVP Timeline**: 3-4 months  
**Target Scale**: 100K+ indexed images, 1K daily searches  

## 2. System Architecture

### 2.1 High-Level Architecture
```
[React Frontend] → [FastAPI Backend] → [CLIP Model Service]
                                            ↓
              [PostgreSQL] ← [Vector Database (Chroma/Pinecone)]
                                            ↓
                             [File Storage (S3/Local)]
```

### 2.2 Core Components

**Frontend Layer**
- **Technology**: React with Tailwind CSS
- **Features**: Image upload (drag/drop), results grid, keyword display
- **Deployment**: Static hosting (Vercel/Netlify)

**API Layer** 
- **Technology**: FastAPI (Python)
- **Endpoints**:
  - `POST /search` - Image upload and similarity search
  - `GET /image/{id}` - Serve indexed images
  - `GET /health` - System health checks
- **Deployment**: Docker container

**ML Service**
- **Model**: CLIP (ViT-B/32) via Transformers library
- **Function**: Extract 512-dimensional feature vectors from images
- **Integration**: Embedded within FastAPI service

**Data Layer**
- **Vector Database**: Chroma (self-hosted) or Pinecone (cloud)
- **Metadata Database**: PostgreSQL
- **File Storage**: Local filesystem → AWS S3 (production)

## 3. Data Flow

### 3.1 Indexing Pipeline (Offline)
1. New images added to file storage
2. Background indexing service processes batches
3. CLIP extracts feature vectors (512-dim)
4. Vectors stored in Chroma, metadata in PostgreSQL
5. Thumbnails generated and stored

### 3.2 Search Pipeline (Real-time)
1. User uploads image via React frontend
2. FastAPI extracts CLIP features from query image
3. Vector similarity search in Chroma (cosine similarity)
4. Fetch metadata for top 20 similar images
5. Generate keywords from result patterns
6. Return JSON response with images, keywords, sources
7. Frontend renders results grid

## 4. Technology Stack

### 4.1 MVP Stack
```yaml
Frontend:
  - React 18
  - Tailwind CSS
  - Axios for API calls

Backend:
  - FastAPI 0.100+
  - Python 3.9+
  - Uvicorn ASGI server

ML/AI:
  - CLIP via HuggingFace Transformers
  - PyTorch 2.0+
  - PIL for image processing

Databases:
  - Chroma (vector database)
  - PostgreSQL 15 (metadata)
  - SQLAlchemy ORM

Storage:
  - Local filesystem (dev)
  - AWS S3 + CloudFront (production)

Deployment:
  - Docker & Docker Compose
  - NGINX reverse proxy
```

### 4.2 Database Schema

**PostgreSQL Tables**
```sql
images (
  id UUID PRIMARY KEY,
  filename VARCHAR,
  original_url TEXT,
  source_domain VARCHAR,
  file_size INTEGER,
  dimensions VARCHAR,
  created_at TIMESTAMP,
  indexed_at TIMESTAMP
)

image_tags (
  image_id UUID,
  tag VARCHAR,
  confidence FLOAT
)
```

**Chroma Vector Store**
- Image embeddings (512 dimensions)
- Linked to PostgreSQL via image UUID

## 5. API Design

### 5.1 Core Endpoints

```python
POST /search
# Request: multipart/form-data with image file
# Response: {
#   "keywords": ["dog", "outdoor", "golden retriever"],
#   "similar_images": [
#     {
#       "id": "uuid",
#       "thumbnail_url": "...",
#       "source_url": "...",
#       "source_domain": "flickr.com",
#       "similarity": 0.94
#     }
#   ]
# }

GET /image/{id}
# Serves image files and thumbnails

GET /health
# System status and model info
```

## 6. Infrastructure & Deployment

### 6.1 MVP Deployment
**Single Server Setup**
- Docker Compose orchestration
- All services on one machine
- NGINX for static files and reverse proxy
- Let's Encrypt for SSL

### 6.2 Production Scaling Path
- **Frontend**: CDN deployment
- **API**: Load balanced containers
- **Vector DB**: Managed Pinecone
- **Storage**: S3 + CloudFront
- **Monitoring**: Basic logging + health checks

## 7. Performance Requirements

### 7.1 Target Metrics
- **Search latency**: <2 seconds end-to-end
- **Indexing throughput**: 1000+ images/hour
- **Concurrent users**: 50+ simultaneous searches
- **Database size**: 100K-1M indexed images

### 7.2 Optimization Strategies
- Keep CLIP model loaded in memory
- Generate thumbnails during indexing
- Implement result caching for popular queries
- Use CDN for static assets

## 8. Development Phases

### Phase 1: Core MVP (Weeks 1-6)
- [ ] Basic React frontend with upload
- [ ] FastAPI backend with CLIP integration
- [ ] Chroma vector database setup
- [ ] Simple similarity search working
- [ ] Local file storage

### Phase 2: Enhancement (Weeks 7-10)
- [ ] Keyword generation from search results
- [ ] Source attribution and metadata
- [ ] Improved UI/UX
- [ ] Basic indexing pipeline
- [ ] PostgreSQL integration

### Phase 3: Production Ready (Weeks 11-14)
- [ ] Docker containerization
- [ ] S3 storage migration
- [ ] Performance optimization
- [ ] Error handling and monitoring
- [ ] Basic admin interface

## 9. Risk Assessment

**Technical Risks**:
- CLIP model memory requirements (2GB+ RAM)
- Vector similarity search performance at scale
- Image storage costs

**Mitigation Strategies**:
- Start with smaller model variants
- Implement efficient batching
- Use image compression and thumbnails
- Plan for cloud storage migration

## 10. Success Metrics

**MVP Success Criteria**:
- [ ] Users can upload images and get relevant results
- [ ] Search results include keywords and source attribution
- [ ] System handles 100+ concurrent searches
- [ ] 95% uptime during testing phase

This architecture provides a solid foundation for building a reverse image search system that can start simple and scale as needed.