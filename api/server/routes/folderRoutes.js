const express = require('express');
const router = express.Router();
const entry = require('../utils/security');
const FolderControllers = require('../controllers/folderController');

router.post('/seed', entry,  FolderControllers.seedSystemFolders);
router.post('/create', entry,  FolderControllers.createNewFolder);
router.get('/project/:projectId', entry,  FolderControllers.getProjectFolders);
router.get('/project-uploads/:projectId', entry,  FolderControllers.getProjecUF);





module.exports = router;