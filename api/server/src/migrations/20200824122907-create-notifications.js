'use strict';
module.exports = {
  up: async (queryInterface, Sequelize) => {
    await queryInterface.createTable('notifications', {
      id: {
        type: Sequelize.UUID,
        primaryKey: true,
        defaultValue: Sequelize.UUIDV1,
      },
      projectId: {
        type: Sequelize.UUID,
        allowNull: false,
        unique: false,
      },
      content: {
        type: Sequelize.STRING,
        allowNull: false,
        unique: false,
      },
      type: {
        type: Sequelize.STRING,
        allowNull: false,
        unique: false
      },
      status: {
        type: Sequelize.STRING,
        allowNull: false,
        unique: false,
        defaultValue: "Unseen"
      },
      createdAt: {
        allowNull: false,
        type: Sequelize.DATE
      },
      updatedAt: {
        allowNull: false,
        type: Sequelize.DATE
      }
    });
  },
  down: async (queryInterface, Sequelize) => {
    await queryInterface.dropTable('notifications');
  }
};