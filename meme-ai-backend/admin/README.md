# 🎭 Meme.AI Admin Panel

Beautiful, modern admin interface for managing memes and tags.

## Features

### 📋 Tag Management Page
- Select any approved meme from dropdown
- View meme image preview
- See ML-generated tags (from BLIP-2 and CLIP)
- See user-added tags
- **Add new custom tags**
- **Remove ML tags** that are incorrect
- **Remove user tags** that aren't relevant
- Save all changes with one click

### 🗑️ Meme Moderation Page
- View all memes in grid layout
- Filter by status (pending/approved/rejected/all)
- See meme stats (views, likes, dislikes, favorites)
- See first 5 tags per meme
- **Approve memes** (pending → approved)
- **Reject memes** (any → rejected)
- **Delete memes permanently** (⚠️ cannot be undone)

## How to Use

### 1. Start Backend Server
```bash
cd meme-ai-backend
python3 -m uvicorn app.main:app --reload
```

### 2. Open Admin Panel

**Option A: Direct file access**
```bash
# Just open the HTML file in your browser
open admin/index.html
```

**Option B: Through backend (recommended)**
```
Navigate to: http://localhost:8000/admin
```

### 3. Configure API URL (if needed)
- Default: `http://localhost:8000`
- Change if backend runs on different port

## Tag Management Workflow

1. Click "Tag Management" tab
2. Select a meme from dropdown
3. Preview appears with current tags
4. **To add tags:**
   - Type tag in "Add New Tag" field
   - Press Enter or click "Add Tag"
   - Tag appears in "User Tags" section

5. **To remove tags:**
   - Click the "×" next to any tag
   - Works for both ML and User tags

6. Click "💾 Save Changes"
7. Tags are updated in database

## Meme Moderation Workflow

1. Click "Meme Moderation" tab
2. Use status filter to narrow down memes
3. **For each meme:**
   - ✓ Approve: Makes meme visible in recommendations
   - ✗ Reject: Hides meme from recommendations
   - 🗑️ Delete: Permanently removes from database

4. Click "🔄 Refresh" to reload memes

## Design

- Clean, modern Apple-inspired UI
- Responsive grid layout
- Color-coded tags:
  - 🟦 Blue: User tags
  - 🟧 Orange: ML tags
  - ⚪ Gray: Generic tags
- Status badges:
  - 🟢 Green: Approved
  - 🟡 Yellow: Pending
  - 🔴 Red: Rejected

## Technical Details

- Pure HTML/CSS/JavaScript (no dependencies!)
- Uses Fetch API for backend communication
- Real-time updates
- Responsive design (works on mobile too)
- Built-in error handling
- Confirmation dialogs for destructive actions

## Keyboard Shortcuts

- **Enter** in "Add New Tag" field → Add tag
- **Escape** → Cancel current operation (in future)

## Tips

- Upload 10-20 memes first to have content to manage
- Review ML tags for accuracy (CLIP/BLIP aren't perfect)
- Add context-specific tags manually ("work stress", "exam season")
- Delete test memes before going to production
- Approve memes in batches for efficiency

## Troubleshooting

**"Failed to load memes"**
- Make sure backend server is running
- Check API URL is correct
- Verify CORS is enabled (already configured)

**"Failed to update tags"**
- Check meme is approved (can't edit pending/rejected)
- Verify backend logs for errors

**Admin panel doesn't load**
- Make sure `admin/index.html` exists
- Try accessing directly via file path

