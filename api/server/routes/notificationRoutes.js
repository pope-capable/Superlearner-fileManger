const express = require("express");
const router = express.Router();
const entry = require("../utils/security");
const NotificationController = require("../controllers/notificationsController");

router.get("/all/:projectId/:type", entry, NotificationController.getNotifications);
router.get("/all_user/:userId/:type", entry, NotificationController.getUserUnseen);


module.exports = router;
