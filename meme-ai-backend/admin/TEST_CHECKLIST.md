# ✅ Admin Panel Test Checklist

Use this to verify everything works before production.

## Setup (5 min)

- [ ] Backend server is running (`./start_admin.sh`)
- [ ] Admin panel opens at http://localhost:8000/admin
- [ ] No console errors in browser (F12 → Console)
- [ ] API URL is set to `http://localhost:8000`

## Tag Management Tests (15 min)

### Basic Functionality
- [ ] Dropdown loads with memes
- [ ] Selecting meme shows image preview
- [ ] ML tags display correctly (orange badges)
- [ ] User tags display correctly (blue badges)

### Add Tags
- [ ] Type tag in input field
- [ ] Press Enter → tag appears in User Tags
- [ ] Click "Add Tag" button → tag appears
- [ ] Duplicate tags are rejected

### Remove Tags
- [ ] Click × on ML tag → tag disappears
- [ ] Click × on User tag → tag disappears
- [ ] Removed tags don't come back on refresh

### Save Changes
- [ ] Click "💾 Save Changes" → success message
- [ ] Refresh page → changes persist
- [ ] Check Firebase console → tags updated
- [ ] Cancel button clears form

## Meme Moderation Tests (20 min)

### View Memes
- [ ] Meme grid loads with images
- [ ] Images display correctly
- [ ] Meme IDs shown
- [ ] Status badges correct (green/yellow/red)
- [ ] Stats display (views, likes, etc.)
- [ ] Tags show (first 5)
- [ ] "+X more" appears if >5 tags

### Filter Memes
- [ ] "All Statuses" shows all memes
- [ ] "Pending" shows only pending
- [ ] "Approved" shows only approved
- [ ] "Rejected" shows only rejected
- [ ] Refresh button reloads grid

### Approve Memes
- [ ] Click "✓ Approve" on pending meme
- [ ] Confirmation dialog appears (optional)
- [ ] Status changes to approved (green badge)
- [ ] Meme disappears if filtered by "Pending"

### Reject Memes
- [ ] Click "✗ Reject" on any meme
- [ ] Status changes to rejected (red badge)
- [ ] Meme still visible in "All" or "Rejected" filter

### Delete Memes
- [ ] Click "🗑️ Delete" on any meme
- [ ] Confirmation dialog appears: "PERMANENTLY DELETE"
- [ ] Click OK → meme disappears from grid
- [ ] Check Firebase → meme is gone
- [ ] ⚠️ THIS CANNOT BE UNDONE (expected behavior)

## Edge Cases (10 min)

### Error Handling
- [ ] Backend offline → "Failed to load" error
- [ ] Invalid API URL → connection error
- [ ] Delete non-existent meme → error message
- [ ] Update tags for deleted meme → error message

### Empty States
- [ ] No memes uploaded → "No memes found"
- [ ] All memes filtered out → empty grid
- [ ] Meme has no tags → empty tag section

### Large Datasets
- [ ] 50+ memes load quickly
- [ ] Scrolling is smooth
- [ ] Images load on-demand (lazy loading works)
- [ ] Grid responsive on resize

## Browser Compatibility (5 min)

- [ ] Chrome/Edge (Chromium)
- [ ] Safari
- [ ] Firefox
- [ ] Mobile Safari (iOS)
- [ ] Mobile Chrome (Android)

## Responsive Design (5 min)

- [ ] Desktop (1920px) → 3 columns
- [ ] Laptop (1440px) → 3 columns
- [ ] Tablet (768px) → 2 columns
- [ ] Mobile (375px) → 1 column
- [ ] All buttons accessible on mobile

## Security & Best Practices (5 min)

- [ ] CORS enabled for admin panel
- [ ] Confirmation for destructive actions
- [ ] No XSS vulnerabilities (test with `<script>` in tags)
- [ ] No console errors or warnings
- [ ] API errors handled gracefully

## Performance (5 min)

- [ ] Page loads in < 2 seconds
- [ ] Meme grid renders in < 1 second
- [ ] Tag updates save in < 500ms
- [ ] No memory leaks (check DevTools)
- [ ] Images optimized/lazy-loaded

## Production Readiness (5 min)

- [ ] All test memes deleted
- [ ] API URL set to production (if deployed)
- [ ] Admin panel password protected (future)
- [ ] HTTPS enabled (future)
- [ ] Logs reviewed for errors

---

## Quick Test Commands (Copy & Paste)

```bash
# Test 1: Create test user
curl -X POST http://localhost:8000/users \
  -H "Content-Type: application/json" \
  -d '{"id": "admin_test", "email": "admin@test.com"}'

# Test 2: Upload test meme
curl -X POST http://localhost:8000/memes \
  -H "Content-Type: application/json" \
  -d '{
    "id": "test_meme_admin",
    "image_url": "https://i.imgflip.com/4t0m5.jpg",
    "user_tags": ["test"]
  }'

# Test 3: Check it's pending
curl http://localhost:8000/memes/test_meme_admin

# Test 4: Approve via API
curl -X POST http://localhost:8000/memes/test_meme_admin/approve

# Test 5: Verify in admin panel
open http://localhost:8000/admin
```

---

## Common Issues & Fixes

### Issue: "Failed to load memes"
**Fix**: Check backend is running on port 8000

### Issue: "CORS error"
**Fix**: Already configured in main.py (allow_origins=["*"])

### Issue: Images not loading
**Fix**: Check image URLs are valid and accessible

### Issue: Tags not saving
**Fix**: Check meme status is "approved" (can't edit pending)

### Issue: Admin panel blank
**Fix**: Check browser console for JS errors

### Issue: Can't delete meme
**Fix**: Verify you clicked "OK" in confirmation dialog

---

## Total Test Time: ~70 minutes

**Quick test**: 20 min (just basic functionality)
**Full test**: 70 min (everything above)

---

## ✅ All Tests Passed?

Congratulations! Your admin panel is production-ready! 🎉

Next steps:
1. Upload 50-100 real memes
2. Review and approve them
3. Add custom tags for better recommendations
4. Start iOS development

