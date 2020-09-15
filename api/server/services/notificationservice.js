const database = require("../src/models");
const Sequelize = require("sequelize");
const Op = Sequelize.Op;

class NotificationServices {
      // Create project Record
  static async createNotification(data) {
    try {
      return await database.notifications.create(data);
    } catch (error) {
      throw error;
    }
  }

    // get one project breakdown
    static async getunseen(data) {
        try {
          return await database.notifications.findAll({
            where: { projectId: data.projectId, type: data.type, status: "Unseen" }
          });
        } catch (error) {
          throw error;
        }
      }

        // get unseen notifications with userId
        static async getUserUnseen(data) {
          try {
            return await database.notifications.findAll({
              where: { userId: data.userId, type: data.type, status: "Unseen" }
            });
          } catch (error) {
            throw error;
          }
        }

}

module.exports = NotificationServices;
