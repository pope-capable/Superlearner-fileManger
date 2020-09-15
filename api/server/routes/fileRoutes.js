const express = require('express');
const router = express.Router();
const entry = require('../utils/security');
// const upload = require('../utils/upload')
const FileController = require('../controllers/fileController');
var multer  = require('multer')
var multerS3 = require('multer-s3')
var aws = require('aws-sdk')

aws.config.update({
    secretAccessKey: 'leJcn4UlL2mVXbSHXA1Ka+f/UeF1E9JLOccYOV7J',
    accessKeyId: 'AKIAJM4B2RBJFKJIXQAA',
});

var s3 = new aws.S3()

var upload = multer({
    storage: multerS3({
    s3: s3,
    bucket: 'superlearner',
    metadata: function (req, file, cb) {
      cb(null, {fieldName: file.fieldname});
    },
    key: function (req, file, cb) {
      cb(null, Date.now().toString())
      var getdate = Date.now().toString()
      var useKey = `${getdate + file.originalname}`
      cb(null, useKey); //use Date.now() for unique file keys

    }
  })
 })

router.post('/upload-one', entry, upload.single("document"), FileController.uploadOneFile);
router.get('/all/:folderId', entry,  FileController.getFolderFiles);
router.get('/recent-res/:projectId', entry, FileController.getRecentFolderFiles)




module.exports = router;